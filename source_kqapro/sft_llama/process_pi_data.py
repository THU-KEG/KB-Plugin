import datasets
from llama import LlamaTokenizerFast
from itertools import chain
import os
import random
import torch
import json
from copy import deepcopy
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--metaqa', type=int, default=0)
args = parser.parse_args()

BS = 16

def get_program_seq(program):
    seq = []
    for x in program:
        func, inputs = x["function"], x["inputs"]
        args = ''
        for input in inputs:
            args += ' <arg> ' + input.lower()
        seq.append(func + args)
    if func in ["SelectBetween", "SelectAmong"]:
        seq.append("What")
    seq = ' <func> '.join(seq)
    
    return seq


model_path = "../../models_hf/llama-2-7b"
tokenizer = LlamaTokenizerFast.from_pretrained(model_path)

def process_function(examples):
    result = {
        "input_ids": [],
        "labels": [],
        "kb_idxs": [], 
    }
    for prompt, program, kb_idx in zip(examples["question"], examples["program"], examples["kb_idx"]):
        output = get_program_seq(program)
        prompt = prompt.lower() + '\n'
        ipt = prompt + ' ' + output
        prompt = torch.LongTensor(tokenizer.encode(prompt))
        input_ids = tokenizer.encode(ipt)
        input_ids.append(tokenizer.eos_token_id)
        input_ids = torch.LongTensor(input_ids)
        assert torch.equal(prompt, input_ids[:len(prompt)])
        labels = deepcopy(input_ids)
        labels[:len(prompt)] = -100
        result["input_ids"].append(input_ids)
        result["labels"].append(labels)
        result["kb_idxs"].append(kb_idx)
    return result

if args.metaqa:
    save_dir = "../../data/kqapro/PI/multi_kb_no_metaqa"
else:
    save_dir = "../../data/kqapro/PI/multi_kb"
    
data_files ={
    "train": f"{save_dir}/train.json",
}

raw_datasets = datasets.load_dataset(
    "json",
    data_files=data_files,
)

lm_datasets = datasets.DatasetDict()
for key in raw_datasets:
    lm_datasets[key] = raw_datasets[key].map(
        process_function,
        batched=True,
        num_proc=64,
        remove_columns=raw_datasets[key].column_names,
    )
    lm_datasets[key] = lm_datasets[key].sort("kb_idxs")
    batch_idxs = [i//BS for i in range(len(lm_datasets[key]))]
    idxs = list(range(len(lm_datasets[key])//BS))
    random.seed(666)
    random.shuffle(idxs)
    lm_datasets[key] = lm_datasets[key].add_column(name="batch_idx", column=[idxs[i] for i in batch_idxs])
    lm_datasets[key] = lm_datasets[key].sort("batch_idx")
    lm_datasets[key] = lm_datasets[key]

os.makedirs(save_dir, exist_ok=True)
lm_datasets.save_to_disk(save_dir)
