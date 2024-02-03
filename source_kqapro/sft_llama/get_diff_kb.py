import json
from tqdm import tqdm
import random
from collections import defaultdict
import os
import copy

random.seed(666)

T = 16

qid2aliases = json.load(open("../../data/kqapro/qid2aliases.json", "r"))
p2aliases = json.load(open("../../data/kqapro/p2aliases.json", "r"))

save_dir = f"../../data/kqapro/diff_kb/0"
os.makedirs(save_dir, exist_ok=True)
os.system(f"cp ../../data/kqapro/kb.json {save_dir}/kb.json")
os.system(f"cp ../../data/kqapro/train.json {save_dir}/train.json")
os.system(f"cp ../../data/kqapro/dev.json {save_dir}/dev.json")
qid2alias = {qid:x["name"] for (qid, x) in qid2aliases.items()}
p2alias = {p:p for p in p2aliases}
json.dump(qid2alias, open(f"{save_dir}/qid2alias.json", "w"), indent=2, ensure_ascii=False)
json.dump(p2alias, open(f"{save_dir}/p2alias.json", "w"), indent=2, ensure_ascii=False)

for t in range(1, T):
    print(t)
    
    save_dir = f"../../data/kqapro/diff_kb/{t}"
    
    os.makedirs(save_dir, exist_ok=True)
    qid2alias, p2alias = {}, {}
    for qid in qid2aliases:
        qid2alias[qid] = random.choice(qid2aliases[qid]["aliases"])
    for p in p2aliases:
        p2alias[p] = random.choice(p2aliases[p])
    json.dump(qid2alias, open(f"{save_dir}/qid2alias.json", "w"), indent=2, ensure_ascii=False)
    json.dump(p2alias, open(f"{save_dir}/p2alias.json", "w"), indent=2, ensure_ascii=False)
    
    kb = json.load(open("../../data/kqapro/kb.json", "r"))
    for qid, entity in tqdm(list(kb["concepts"].items())):
        entity["name"] = qid2alias[qid]
        
    for qid, entity in tqdm(list(kb["entities"].items())):
        entity["name"] = qid2alias[qid]
        for attribute in entity["attributes"]:
            attribute["key"] = p2alias[attribute["key"]]
            qualifiers = defaultdict(list)
            for qualifier, value in attribute["qualifiers"].items():
                qualifiers[p2alias[qualifier]].extend(value)
            attribute["qualifiers"] = qualifiers
            
        for relation in entity["relations"]:
            relation["relation"] = p2alias[relation["relation"]]
            qualifiers = defaultdict(list)
            for qualifier, value in relation["qualifiers"].items():
                qualifiers[p2alias[qualifier]].extend(value)
            relation["qualifiers"] = qualifiers
    json.dump(kb, open(f"{save_dir}/kb.json", "w"), indent=2, ensure_ascii=False)
    
    # qid2alias = json.load(open(f"{save_dir}/qid2alias.json", "r"))
    # p2alias = json.load(open(f"{save_dir}/p2alias.json", "r"))
    # kb = json.load(open(f"{save_dir}/kb.json", "r"))
    
    for split in ["dev", "train"]:
        data = json.load(open(f"../../data/kqapro/linked_{split}.json"))
        new_data = []
        for datum in tqdm(data):
            question = datum["question"]
            linked_program = copy.deepcopy(datum["linked_program"])
            program = []
            for x in linked_program:
                func, inputs = x["function"], x["inputs"]
                
                if func == "Find":
                    question = question.replace(qid2aliases[inputs[0]]["name"], qid2alias[inputs[0]])
                    
                if func in ["Find", "FilterConcept"]:
                    inputs = [qid2alias[inputs[0]]]
                elif func in ["FilterStr", "FilterNum", "FilterYear", "FilterDate"]:
                    inputs[0] = p2alias[inputs[0]]
                elif func in ["QFilterStr", "QFilterNum", "QFilterYear", "QFilterDate"]:
                    inputs[0] = p2alias[inputs[0]]
                elif func == "Relate":
                    inputs[0] = p2alias[inputs[0]]
                elif func in ["SelectBetween", "SelectAmong"]:
                    inputs[0] = p2alias[inputs[0]]
                elif func in ["QueryAttr"]:
                    inputs[0] = p2alias[inputs[0]]
                elif func in ["QueryAttrUnderCondition"]:
                    inputs[0] = p2alias[inputs[0]]
                    inputs[1] = p2alias[inputs[1]]
                elif func in ["QueryAttrQualifier"]:
                    inputs[0] = p2alias[inputs[0]]
                    inputs[2] = p2alias[inputs[2]]
                elif func in ["QueryRelationQualifier"]:
                    inputs[0] = p2alias[inputs[0]]
                    inputs[1] = p2alias[inputs[1]]   
                program.append({
                    "function": func,
                    "inputs": inputs
                }) 
                
            if program[-1]["function"] in ["What", "SelectBetween", "SelectAmong"]:
                answer = qid2alias[datum["linked_ans"]] 
            elif program[-1]["function"] == "QueryRelation":
                answer = p2alias[datum["answer"]]
            else:
                answer = datum["answer"]
                
            new_data.append({
                "question": question,
                "answer": answer,
                "program": program,
            })
        json.dump(new_data, open(f"{save_dir}/{split}.json", "w"), indent=2, ensure_ascii=False)
    
            