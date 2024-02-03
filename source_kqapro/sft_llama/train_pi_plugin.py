#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import copy
import logging
from dataclasses import dataclass, field
from typing import Optional, Dict, Sequence, List
import os, sys
import time
import numpy as np
import torch
import transformers
from torch.utils.data import Dataset
from transformers import Seq2SeqTrainer, DataCollatorForSeq2Seq, TrainerCallback, GenerationConfig, set_seed, TrainerState, TrainerControl
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
from llama import LlamaForCausalLM, LlamaTokenizer
from peft import ( 
    LoraConfig,
    PeftModel,
    get_peft_model,
    get_peft_model_state_dict,
    set_peft_model_state_dict,
)
import datasets
import json
from args import TrainingArguments, ModelArguments, DataArguments
from tqdm import tqdm
import shutil
from pi_plugin_trainer import PIPluginTrainer

logger = logging.getLogger(__name__)

def setup_logging(training_args):
    os.makedirs(training_args.logging_dir, exist_ok=True)
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)

    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16 or training_args.bf16}"
    )
    
IGNORE_INDEX = -100
    
@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels, kb_idxs = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels", "kb_idxs"))
        assert all(i == kb_idxs[0] for i in kb_idxs), kb_idxs
        for i in range(len(input_ids)):
            input_ids[i] = torch.tensor(input_ids[i])
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        for i in range(len(labels)):
            labels[i] = torch.tensor(labels[i])
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)
        return dict(
            input_ids=input_ids,
            labels=labels,
            kb_idx=str(kb_idxs[0]),
        )

def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer, data_args, training_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    logger.info("Loading data...")
    raw_datasets = datasets.load_from_disk(data_args.processed_data_dir)

    train_dataset = raw_datasets["train"]
    eval_dataset = raw_datasets.get("validation")
    if data_args.max_train_samples is not None:
        max_train_samples = min(len(train_dataset), data_args.max_train_samples)
        train_dataset = train_dataset.select(range(max_train_samples))
    if eval_dataset is not None and data_args.max_eval_samples is not None:
        max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
        eval_dataset = eval_dataset.select(range(max_eval_samples))

    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    schema_plugins = {}
    for key, path in json.load(open(data_args.schema_plugin_path)).items():
        print(f"load {key}: {path}")
        schema_plugins[key] = {
            "config": LoraConfig.from_pretrained(path),
            "state_dict": torch.load(f"{path}/adapter_model.bin", map_location="cpu")
        }
    
    return dict(schema_plugins=schema_plugins, train_dataset=train_dataset, eval_dataset=eval_dataset, data_collator=data_collator)

class SavePeftModelCallback(TrainerCallback):
    def on_save(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        if state.is_world_process_zero:
            print('+++++++++++++++++save call back++++++++++++++++')
            checkpoint_folder = os.path.join(
                args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}"
            )
            kwargs["model"].save_pretrained(checkpoint_folder)

            pytorch_model_path = os.path.join(checkpoint_folder, "pytorch_model.bin")
            if os.path.exists(pytorch_model_path):
                os.remove(pytorch_model_path)
            optimizer_state_path = os.path.join(checkpoint_folder, f"global_step{state.global_step}")
            if os.path.exists(optimizer_state_path):
                shutil.rmtree(optimizer_state_path)
            return control


def train():
    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    
    setup_logging(training_args)
    set_seed(training_args.seed)
    
    tokenizer = LlamaTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
    )
    tokenizer.pad_token = tokenizer.eos_token
        
    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args, training_args=training_args)
    
    lora_config = LoraConfig(
        r=model_args.lora_r,
        lora_alpha=model_args.lora_alpha,
        target_modules =  model_args.target_modules,
        fan_in_fan_out = False,
        lora_dropout=0.05,
        inference_mode=False,
        bias="none",
        task_type="CAUSAL_LM",
    )
    print(lora_config)
    
    model = LlamaForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
    )
    model.enable_input_require_grads()
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    trainer = PIPluginTrainer(
        model=model, 
        tokenizer=tokenizer, 
        args=training_args, 
        # compute_metrics=compute_metrics,
        callbacks=[SavePeftModelCallback],
        **data_module
    )
    
    if local_rank == 0:
        os.makedirs("output_config", exist_ok=True)
        with open('output_config/args.json', 'w') as fout:
            fout.write(training_args.to_json_string())
        if trainer.args.deepspeed:
            json.dump(trainer.args.hf_deepspeed_config.config, open('output_config/deepspeed.json', 'w'), indent=4)
            
    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)
        
    trainer.train()
    trainer.save_state()


if __name__ == "__main__":
    train()