import json
from tqdm import tqdm
from kopl.kopl2 import KoPLEngine
import os

executor = KoPLEngine(json.load(open("data/kb.json")))
for split in ["dev", "train"]:
    correct, count = 0, 0
    data = json.load(open(f"data/{split}.json"))
    new_data = []
    for datum in tqdm(data):
        golden_ans, program = datum["answer"], datum["program"]
        func_list, inputs_list = [x["function"] for x in program], [x["inputs"] for x in program]
        results = executor.forward(func_list, inputs_list, ignore_error=True)
        for ans, (func_list, inputs_list) in results:
            if ans is None:
                continue
            else:
                if isinstance(ans, list):
                    if len(ans) == 0:
                        continue
                    ans = ans[0]
                        
                ans_qid = None 
                if func_list[-1] in ["What", "SelectBetween", "SelectAmong"]:
                    ans_qid = ans
                    ans = executor.kb.entities[ans_qid]['name']
                if ans == golden_ans:
                    correct += 1
                    datum["linked_program"] = [{"function": x, "inputs": y} for (x, y) in zip(func_list, inputs_list)]
                    datum["linked_ans"] = ans_qid if ans_qid is not None else ans
                    new_data.append(datum)
                    break
        count += 1
    acc = correct / count
    print(split)
    print(acc)
    print(len(data), len(new_data))
    json.dump(new_data, open(f"data/linked_{split}.json", "w"), indent=2, ensure_ascii=False)
    
        
