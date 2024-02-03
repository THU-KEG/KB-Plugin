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
parser.add_argument('--dataset', type=str, required=True)
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
        if len(input_ids) > 500:
            # print(len(input_ids))
            # print(ipt)
            # exit()
            continue
        input_ids.append(tokenizer.eos_token_id)
        input_ids = torch.LongTensor(input_ids)
        assert torch.equal(prompt, input_ids[:len(prompt)])
        labels = deepcopy(input_ids)
        labels[:len(prompt)] = -100
        result["input_ids"].append(input_ids)
        result["labels"].append(labels)
    return result

data_path = f"../../data/{args.dataset}/hrt.json"
data_files ={
    "train": data_path
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

save_dir = data_path[:-5]
os.makedirs(save_dir, exist_ok=True)
lm_datasets.save_to_disk(save_dir)