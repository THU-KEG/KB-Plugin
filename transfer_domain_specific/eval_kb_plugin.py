from model.llama import LlamaForCausalLM, LlamaTokenizer, LlamaConfig
from model.kopl_constrained_decoder import KoPLConstrainedDecoder
from peft import PeftModel, PeftConfig, LoraConfig
import torch
import json
from torch.utils.data import Dataset, DataLoader
from kopl.kopl import KoPLEngine
from tqdm import tqdm
from transformers import set_seed
from dataset import KoPLDataset, collate_fn
from functools import partial
import numpy as np
import random
import os
import spacy
import argparse
from collections import defaultdict

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)
    
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
        lora_A_key = ".".join(name.split(".")[:-1]) + ".lora_A." + name.split(".")[-1]
        if not isinstance(model, PeftModel):
            lora_A_key = "base_model.model." + lora_A_key
        if lora_A_key not in adapter_weights:
            continue
        cnt += 2
        lora_B_key = lora_A_key.replace("lora_A", "lora_B")
        lora_A, lora_B = adapter_weights[lora_A_key].to(p.device).type_as(p.data), adapter_weights[lora_B_key].type_as(p.data).to(p.device)
        p.data += transpose(lora_B @ lora_A, fan_in_fan_out) * scaling
    assert cnt == len(adapter_weights), "Some LORA weights are not used."
    
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, required=True)
args = parser.parse_args()

if args.dataset == "metaqa":
    _dataset = "metaqa"
    data_path = "../data/metaqa/test.json"
    schema_plugin_path = "../checkpoints/metaqa/schema_plugin/train_el_lora_3epoch/checkpoint-706"
    pi_plugin_path = "../checkpoints/kqapro/pi_plugin/train_pi_plugin_no_metaqa_bs16_accu16_kb-epoch3/checkpoint-5000"
elif args.dataset == "soaybench":
    _dataset = "soay"
    data_path = "../data/soaybench/test.jsonl"
    schema_plugin_path = "../checkpoints/soaybench/schema_plugin/train_el_lora_3epoch/checkpoint-477"
    pi_plugin_path = "../checkpoints/kqapro/pi_plugin/train_pi_plugin_bs16_accu16_kb-epoch3/checkpoint-5000"
print(schema_plugin_path)
schema_plugin = {
    "config": LoraConfig.from_pretrained(schema_plugin_path),
    "state_dict": torch.load(f"{schema_plugin_path}/adapter_model.bin", map_location="cuda")
}
print(pi_plugin_path)
pi_plugin = {
    "config": LoraConfig.from_pretrained(pi_plugin_path),
    "state_dict": torch.load(f"{pi_plugin_path}/adapter_model.bin", map_location="cuda")
}

config = pi_plugin["config"]
base_model_name_or_path = "../models_hf/llama-2-7b"
tokenizer = LlamaTokenizer.from_pretrained(base_model_name_or_path, padding_side="left")
tokenizer.pad_token = tokenizer.eos_token

eval_dataset = KoPLDataset(
    dataset=_dataset,
    data_path=data_path,
    perfect_el=False if _dataset == 'soay' else True,
    )
eval_loader = DataLoader(eval_dataset, batch_size=1, collate_fn=partial(collate_fn, tokenizer))

model = KoPLConstrainedDecoder.from_pretrained(base_model_name_or_path, dataset=_dataset, tokenizer=tokenizer, device_map='cuda').bfloat16()
if _dataset == "soay":
    model.spacy_ner = spacy.load("en_core_web_sm")
model.eval()

merge_lora(model, schema_plugin)
merge_lora(model, pi_plugin)

num_beams = 5
max_new_length = 128

with torch.no_grad():
    all_inputs = []
    all_results = []
    f1_sum, em_sum, total = 0, 0, 0
    pbar = tqdm(total=len(eval_loader))
    for idx, batch in enumerate(eval_loader):
        seed_everything(666)
        print()
        print(batch["qid"][0])
        print(batch["question"][0])
        gen_kwargs = {
            "topic_entities": batch["topic_entities"],
            "question": batch["question"],
            "max_new_tokens": max_new_length, 
            "eos_token_id": tokenizer.eos_token_id, 
            "num_beams": num_beams,
            "num_return_sequences": num_beams,
        }
        if _dataset == "metaqa":
            gen_kwargs["expected_hop"] = batch["expected_hop"]
        output_programs = model.generate(**batch["inputs"].to("cuda"), **gen_kwargs)
        assert len(output_programs) == len(batch["question"])
        for i in range(len(output_programs)):
            total += 1
            program = output_programs[i]
            result = {key:batch[key][i] for key in batch if key != "inputs"}
            if isinstance(program, str):
                assert "Exceed Max Length" in program or "EMPTY:" in program or "API ERROR" in program, program
                print(program)
                result.update(pred_code=program, prediction=[], em=False, f1=0)
            else:
                pred = set(program.execution)
                answer = set(batch["answer"][i])
                if len(pred.intersection(answer)) != 0:
                    precision = len(pred.intersection(answer)) / len(pred)
                    recall = len(pred.intersection(answer)) / len(answer)
                    f1 = (2 * recall * precision / (recall + precision))
                elif len(answer) == 0 and len(pred) == 0:
                    f1 = 1
                else:
                    f1 = 0
                f1_sum += f1
                em_sum += f1 == 1
                pred, answer = sorted(pred), sorted(answer)
                print(program.code)
                print(pred)
                print(answer)
                result.update(pred_topic_entities=program.topic_entities, answer=answer, pred_code=program.code, prediction=pred, em=f1==1, f1=f1)
            
            pbar.set_description("EM: %.3f F1: %.3f" % (em_sum/total, f1_sum/total))
            pbar.update(1)
            
            all_results.append(result)
        # if idx == 10: break
    
    avg_f1 = f1_sum / total
    avg_em = em_sum / total
    print(max_new_length)
    print(schema_plugin_path)
    print(pi_plugin_path)
    print(avg_f1)
    print(avg_em)
    
    if _dataset == "metaqa":
        hit1 = defaultdict(float)
        cnt = defaultdict(float)
        data = json.load(open("../data/metaqa/test.json"))
        for item, item2 in zip(all_results, data):
            hop = item["qid"].split("_")[0]
            answers, pred = set(item2["correct_answer"]), set(item["prediction"])
            if len(answers & pred) == len(answers) and len(answers) == len(pred):
                hit1[hop] += 1
            else:
                if len(answers & pred) > 0:
                    hit1[hop] += len(answers & pred) / len(answers)
            cnt[hop] += 1
        for hop in hit1:
            print(hop, hit1[hop]/cnt[hop])
            hit1[hop] = hit1[hop]/cnt[hop]
        all_results.append(hit1)
        
    all_results.append({"f1": avg_f1, "em": avg_em, "schema_plugin": schema_plugin_path, "pi_plugin": pi_plugin_path, "num_beams": num_beams, "max_new_length": max_new_length})
    
os.makedirs('./results', exist_ok=True)
json.dump(all_results, open("./results/prediction_%s.json" % (_dataset), "w"), indent=2, ensure_ascii=False)