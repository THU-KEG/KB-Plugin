from torch.utils.data import Dataset, DataLoader
import json
from pathlib import Path
from collections import defaultdict
import jsonlines
import os

path = str(Path(__file__).parent.absolute())

class KoPLDataset(Dataset):
    def __init__(
        self,
        dataset: str = 'metaqa',
        data_path: str = None,
        perfect_el: bool = True,
    ):
        self._dataset = dataset
        self._perfect_el = perfect_el
        self.data = []
        
        if dataset == 'soay':
            with jsonlines.open(data_path, "r") as f:
                for j, item in enumerate(f):
                    question = item["Query_en"]
                    answer = item["Answer"]
                    if isinstance(answer, str) or isinstance(answer, int):
                        answer = [str(answer)]
                    elif isinstance(answer, list):
                        answer = [str(x) for x in answer]
                    elif isinstance(answer, dict):
                        answer = [answer['name']] if 'name' in answer else [answer["info"]["name"]]
                    else:
                        raise NotImplementedError
                    if perfect_el:
                        keys = [x.strip() for x in item["Inputs"].split(',')]
                        topic_entities = [item["Entity_Information"][key] for key in keys]
                    else:
                        topic_entities = None
                    self.data.append({
                        "qid": item["qid"],
                        "question": question,
                        "topic_entities": topic_entities,
                        "target_program": item["Base_Question_zh"],
                        "answer": answer,
                    })
                # exit()
        elif dataset == 'metaqa':
            assert perfect_el
            for item in json.load(open(data_path)):
                item["expected_hop"] = int(item["qid"].split("hop")[0])
                item["answer"] = item["correct_answer"]
                self.data.append(item)
                    
    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)

def collate_fn(tokenizer, batch):
    inputs = tokenizer([(datum["question"] + '\n') for datum in batch], padding=True, return_tensors="pt")
    res = {"inputs": inputs}
    for key in batch[0]:
        res[key] = [datum[key] for datum in batch]
    return res
