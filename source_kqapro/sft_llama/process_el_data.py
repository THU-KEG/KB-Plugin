import datasets
from llama import LlamaTokenizerFast
from itertools import chain
import os
import random
import torch
import json
from copy import deepcopy
from copy import deepcopy
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--metaqa', type=int, default=0)
args = parser.parse_args()

model_path = "../../models_hf/llama-2-7b"
tokenizer = LlamaTokenizerFast.from_pretrained(model_path)

def process_function(examples):
    result = {
        "input_ids": [],
        "labels": []
    }
    for prompt, output in zip(examples["program"], examples["answer"]):
        prompt += '\n'
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
    return result


T = 16
for t in range(T):
    print(t)
    data_files ={
        "train": f"../../data/kqapro/diff_kb/{t}/" + ("hrt_no_metaqa.json" if args.metaqa else "hrt.json")
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
    
    save_dir = f"../../data/kqapro/diff_kb/{t}/" + ("hrt_no_metaqa" if args.metaqa else "hrt")
    os.makedirs(save_dir, exist_ok=True)
    lm_datasets.save_to_disk(save_dir)