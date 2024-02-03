from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union

import torch
from torch import nn
from torch.utils.data import Dataset, SequentialSampler
from transformers import Seq2SeqTrainer
from transformers.utils import is_apex_available, is_datasets_available, is_sagemaker_mp_enabled
from peft import PeftModel, PeftConfig, LoraConfig
if is_apex_available():
    from apex import amp
if is_sagemaker_mp_enabled():
    from transformers.trainer_pt_utils import smp_forward_backward, smp_forward_only, smp_gather, smp_nested_concat
import time

def transpose(weight, fan_in_fan_out):
    return weight.T if fan_in_fan_out else weight

def merge_lora(model, adapter):
    adapter_config, adapter_weights = adapter["config"], adapter["state_dict"]
    for key in adapter_weights:
        assert key.endswith(".weight"), key
    scaling = adapter_config.lora_alpha / adapter_config.r
    fan_in_fan_out = adapter_config.fan_in_fan_out
    cnt = 0
    for name, p in model.named_parameters():
        lora_A_key = ".".join(name.split(".")[1:-1]) + ".lora_A." + name.split(".")[-1]
        if not isinstance(model.module, PeftModel):
            lora_A_key = "base_model.model." + lora_A_key
        if lora_A_key not in adapter_weights:
            continue
        cnt += 2
        lora_B_key = lora_A_key.replace("lora_A", "lora_B")
        #lora_A, lora_B = adapter_weights[lora_A_key].to(p.device).type_as(p.data), adapter_weights[lora_B_key].type_as(p.data).to(p.device)
        lora_A, lora_B = adapter_weights[lora_A_key], adapter_weights[lora_B_key]
        p.data += transpose(lora_B @ lora_A, fan_in_fan_out) * scaling
    assert cnt == len(adapter_weights), "Some LORA weights are not used."
        
def unmerge_lora(model, adapter):
    adapter_config, adapter_weights = adapter["config"], adapter["state_dict"]
    for key in adapter_weights:
        assert key.endswith(".weight"), key
    scaling = adapter_config.lora_alpha / adapter_config.r
    fan_in_fan_out = adapter_config.fan_in_fan_out
    cnt = 0
    for name, p in model.named_parameters():
        lora_A_key = ".".join(name.split(".")[1:-1]) + ".lora_A." + name.split(".")[-1]
        if not isinstance(model.module, PeftModel):
            lora_A_key = "base_model.model." + lora_A_key
        if lora_A_key not in adapter_weights:
            continue
        cnt += 2
        lora_B_key = lora_A_key.replace("lora_A", "lora_B")
        #lora_A, lora_B = adapter_weights[lora_A_key].to(p.device).type_as(p.data), adapter_weights[lora_B_key].type_as(p.data).to(p.device)
        lora_A, lora_B = adapter_weights[lora_A_key], adapter_weights[lora_B_key]
        p.data -= transpose(lora_B @ lora_A, fan_in_fan_out) * scaling
    assert cnt == len(adapter_weights), "Some LORA weight are not used."

class PIPluginTrainer(Seq2SeqTrainer):
    def __init__(self, model, schema_plugins, args, *more_args, **kwargs):
        assert args.remove_unused_columns == False
        super().__init__(model, args, *more_args, **kwargs)
        device = args.device
        dtype = torch.bfloat16 if args.bf16 else (torch.float16 if args.fp16 else torch.float32)
        for kb_idx in schema_plugins:
            for n in schema_plugins[kb_idx]["state_dict"]:
                schema_plugins[kb_idx]["state_dict"][n] = schema_plugins[kb_idx]["state_dict"][n].type(dtype).to(device)
        self.schema_plugins = schema_plugins
                
        
    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
        """
        Perform a training step on a batch of inputs.

        Subclass and override to inject custom behavior.

        Args:
            model (`nn.Module`):
                The model to train.
            inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.

        Return:
            `torch.Tensor`: The tensor with training loss on this batch.
        """
        kb_idx = inputs.pop("kb_idx") 
        kb_plugin = self.schema_plugins[kb_idx]
        
        st = time.time()
        merge_lora(model, kb_plugin)
        cost = time.time() - st
        print(f"merge lora {kb_idx}, {cost} seconds")
        
        model.train()
        inputs = self._prepare_inputs(inputs)

        if is_sagemaker_mp_enabled():
            loss_mb = smp_forward_backward(model, inputs, self.args.gradient_accumulation_steps)
            return loss_mb.reduce_mean().detach().to(self.args.device)

        with self.compute_loss_context_manager():
            loss = self.compute_loss(model, inputs)

        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training

        if self.do_grad_scaling:
            self.scaler.scale(loss).backward()
        elif self.use_apex:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            self.accelerator.backward(loss)

        st = time.time()
        unmerge_lora(model, kb_plugin)
        cost = time.time() - st
        print(f"unmerge lora {kb_idx}, {cost} seconds")
        
        return loss.detach() / self.args.gradient_accumulation_steps
    
    def _get_train_sampler(self) -> Optional[torch.utils.data.Sampler]:
        return SequentialSampler(self.train_dataset)