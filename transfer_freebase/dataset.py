from torch.utils.data import Dataset, DataLoader
import json
from pathlib import Path
from collections import defaultdict
from utils.logic_form_util import get_program_seq

path = str(Path(__file__).parent.absolute())

class KoPLDataset(Dataset):
    def __init__(
        self,
        dataset: str = 'grail',
        data_path: str = None,
        perfect_el: bool = True,
    ):
        self._dataset = dataset
        self._perfect_el = perfect_el
        
        if self._dataset == 'grail':
            with open(path + "/el_results/grail_combined_tiara.json") as f:
                self._el_results = json.load(f)
        elif self._dataset == 'gq1':
            with open(path + "/el_results/graphq_test.json") as f:
                self._el_results = json.load(f)
        elif self._dataset == 'webq':
            with open(path + "/el_results/webqsp_test_elq.json") as f:
                self._el_results = json.load(f)
            
        if self._dataset in ["grail", 'gq1']:
            self._answer_types = defaultdict(lambda: [])
            if self._dataset == "grail":
                at_fn = "answer_types_grail_combined.txt"
            else:
                at_fn = "answer_types_gq1.txt"
            with open(path + f"/answer_typing/{at_fn}", 'r') as f:
                for line in f:
                    line = line.replace("\n", '')
                    fields = line.split('\t')
                    for item in fields[1:]:
                        self._answer_types[fields[0]].append(item)
        
        self.data = []
        data = json.load(open(data_path, "r"))
        for item in data:#[:100]:
            qid = str(item["qid"])
            entity_name = {}
            answer_types = []
            if self._perfect_el:
                if dataset in ["grail", "gq1"]:
                    for node in item["graph_query"]["nodes"]:
                        if node["node_type"] == "entity":
                            assert node["function"] == "none", qid
                            entity_name[node['id']] = node['friendly_name'].lower()
                        elif node["node_type"] == "literal":
                            if node["function"] not in ["argmax", "argmin"]:
                                assert '^^' in node['id'], qid
                                entity_name[node['id']] = node['id'].split('^^')[0].lower()
                        elif node["node_type"] == "class":
                            answer_types.append(node["id"])
                else:
                    for node in item["graph_query"]["nodes"]:
                        if node["node_type"] == "entity":
                            assert node["function"] == "none", qid
                            entity_name[node['id']] = node['friendly_name'].lower()
                    el_results_item = self._el_results[qid]
                    for m in el_results_item:
                        if m not in entity_name:
                            for mid in el_results_item[m]:
                                entity_name[str(mid)] = m.lower()
            else:
                el_results_item = self._el_results[qid]
                for m in el_results_item:
                    for mid in el_results_item[m]:
                        entity_name[str(mid)] = m.lower()
                answer_types = self._answer_types[qid] if dataset != 'webq' else []
            
            if dataset == 'grail':
                question = item["question"]
            elif dataset == 'gq1':
                question = item["question"]
            elif dataset == 'webq':
                question = item["question"] + '?'
            
            self.data.append({
                "qid": qid,
                "level": item.get("level"),
                "question": question,
                "entity_name": entity_name,
                "answer_types": answer_types,
                "target_program": get_program_seq(item["kopl"]) if "kopl" in item else item.get("s_expression"),
                "answer": [x["answer_argument"] for x in item.get("answer", [])],
            })
                    
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