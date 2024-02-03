import json
import os
from tqdm import tqdm

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--metaqa', type=int, default=0)
args = parser.parse_args()

kbs = []
kb_idxs = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
BS = 16
kb_idxs = kb_idxs[:16]

if args.metaqa:
    save_dir = "../../data/kqapro/PI/multi_kb_no_metaqa"
else: 
    save_dir = "../../data/kqapro/PI/multi_kb"
os.makedirs(save_dir, exist_ok=True)

metaqa_r = {"director", "screenwriter", "cast member", "original language of film or TV show", "genre", "publication date"}

def filter_program(program, forbidden_r):
    if len(forbidden_r) == 0:
        return False
    rs = set()
    for x in program:
        func, inputs = x["function"], x["inputs"]
        if func in ["FilterStr", "FilterNum", "FilterYear", "FilterDate"]:
            rs.add(inputs[0])
        elif func in ["QFilterStr", "QFilterNum", "QFilterYear", "QFilterDate"]:
            rs.add(inputs[0])
        elif func == "Relate":
            rs.add(inputs[0])
        elif func in ["SelectBetween", "SelectAmong"]:
            rs.add(inputs[0])
        elif func in ["QueryAttr"]:
            rs.add(inputs[0])
        elif func in ["QueryAttrUnderCondition"]:
            rs.add(inputs[0])
            rs.add(inputs[1])
        elif func in ["QueryAttrQualifier"]:
            rs.add(inputs[0])
            rs.add(inputs[2])
        elif func in ["QueryRelationQualifier"]:
            rs.add(inputs[0])
            rs.add(inputs[1])
    if len(rs & forbidden_r) > 0:
        return True
    return False

filter_qids = set()
for split in ["train"]:
    new_data = []
    for kb_idx in kb_idxs:
        print(kb_idx)
        data = json.load(open(f"../../data/kqapro/diff_kb/{kb_idx}/{split}.json", "r"))
        p2alias = json.load(open(f"../../data/kqapro/diff_kb/{kb_idx}/p2alias.json"))
        if args.metaqa:
            forbidden_r = {p2alias[r] for r in metaqa_r}
        else:
            forbidden_r = {}
        print(len(data))
        data = [x for qid, x in enumerate(data) if not filter_program(x["program"], forbidden_r)]
        print(len(data))
        data = data[:len(data)//BS*BS]
        print(len(data))
        for datum in tqdm(data):
            new_data.append({
                "question": datum["question"],
                "answer": datum["answer"],
                "program": datum["program"],
                "kb_idx": kb_idx,
            })
    print(len(new_data))
    json.dump(new_data, open(f"{save_dir}/{split}.json", "w"), indent=2, ensure_ascii=False)

schema_plugin_paths = {}
for kb_idx in kb_idxs:
    if args.metaqa:
        schema_plugin_path = f"../../checkpoints/kqapro/schema_plugin/train_el_lora_{kb_idx}_no_metaqa_3epoch"
    else:
        schema_plugin_path = f"../../checkpoints/kqapro/schema_plugin/train_el_lora_{kb_idx}_3epoch"
    for file_name in os.listdir(schema_plugin_path):
        if "checkpoint-18" in file_name:
            schema_plugin_path += "/" + file_name
            break
    schema_plugin_paths[str(kb_idx)] = schema_plugin_path
json.dump(schema_plugin_paths, open(f"{save_dir}/schema_plugin_paths_3epoch.json", "w"), indent=2)



        