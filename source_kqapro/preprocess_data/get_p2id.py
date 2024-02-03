import json
from tqdm import tqdm
import csv

rel2id = {}
# id2rel = json.load(open("data/p2label.json", "r"))
# for r_id, rel in id2rel.items():
#     rel2id[rel] = r_id

with open('data/property.csv', 'r') as file:
    reader = csv.reader(file)
    reader.__next__()
    for id, rel, desc in reader:
        rel2id[rel] = id
kb = json.load(open("data/kb.json", "r"))
rels = set()
for h_id, entity in tqdm(list(kb["entities"].items())):
    for attribute in entity["attributes"]:
        rels.add(attribute["key"])
        for qualifier in attribute["qualifiers"]:
            rels.add(qualifier)
    for relation in entity["relations"]:
        rels.add(relation["relation"])
        for qualifier in relation["qualifiers"]:
            rels.add(qualifier)
            
rels = list(rels)
print(len(rels))
cnt = 0
p2id = {}
for rel in rels:
    if rel not in rel2id:
        print(rel)
        cnt += 1
    else:
        p2id[rel] = "P"+rel2id[rel]
for rel in rels:
    if rel not in rel2id:
        p2id[rel] = ""
print(cnt)
json.dump(p2id, open("data/p2id.json", "w"), indent=2, ensure_ascii=False)