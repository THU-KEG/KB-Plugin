import json
from tqdm import tqdm
from collections import defaultdict
import os
import random
from copy import deepcopy
random.seed(666)


def get_program_seq(program):
    seq = []
    for func, inputs in program:
        args = ''
        for input in inputs:
            args += ' | ' + input
        seq.append(func + args)
    seq = ' <func> '.join(seq)
    return seq

def get_value(t):
    if t["type"] == "quantity":
        return (str(t["value"]) + " " + t["unit"]) if t["unit"] != "1" else str(t["value"])
    else:
        return str(t["value"])

max_size = 3000
min_size = 500

save_dir = f"../../data/metaqa"

data = []
dataset = set()
kb = json.load(open(f"{save_dir}/kb.json", "r"))

def get_concept(x):
    if isinstance(x, str):
        if x in kb["entities"]:
            cid = kb["entities"][x]["instanceOf"][0]
            return kb["concepts"][cid]['name']
        else:
            return kb["concepts"][x]['name']
    else:
        if x["type"] == "quantity":
            return "float" if "." in str(t["value"]) else "int"
        else:
            return x["type"]

def get_name(x):
    if x in kb["entities"]:
        return kb["entities"][x]["name"]
    else:
        return kb["concepts"][x]['name']
    
relation2facts = defaultdict(list)
attribute2facts = defaultdict(list)
cid2eids = defaultdict(list)
for qid, entity in tqdm(list(kb["entities"].items())):
    hid = qid
    assert len(entity['instanceOf']) == 1
    cid = entity['instanceOf'][0]
    cid2eids[cid].append(hid)
    for attribute in entity["attributes"]:
        at = attribute["key"]
        t = attribute["value"]
        qualifiers = attribute["qualifiers"]
        attribute2facts[at].append([hid, at, t, qualifiers])
        
    for relation in entity["relations"]:
        r = relation["relation"] 
        tid = relation["object"]
        qualifiers = relation["qualifiers"]
        if relation["direction"] == "forward":
            relation2facts[r].append([hid, r, tid, qualifiers])

def sample_facts(facts, repeat=False):
    random.shuffle(facts)
    facts = facts[:max_size]
    if repeat:
        if len(facts) < min_size:
            a, b = min_size // len(facts), min_size % len(facts)
            facts = facts * a + facts[:b]
    return facts

def add_relation(hid, r, tid, qualifiers):
    h, t = get_name(hid), get_name(tid)
    hc, tc = get_concept(hid), get_concept(tid)
    for qkey, values in qualifiers.items():
        qualifiers[qkey] = (get_concept(values[0]), get_value(values[0]))
        
    # h r q v -> t
    program = [("Relate", [h, hc, r, "forward"])]
    for qkey, (qc, qv) in qualifiers.items():
        program.append(("FilterQualifier", [qkey, qv]))
    program = get_program_seq(program)
    data.append((program, f'{tc} | {t}'))
    
    # t r q v -> h
    program = [("Relate", [t, tc, r, "backward"])]
    for qkey, (qc, qv) in qualifiers.items():
        program.append(("FilterQualifier", [qkey, qv]))
    program = get_program_seq(program)
    data.append((program, f'{hc} | {h}'))
    
    # h r t q -> v
    for qkey, (qc, qv) in qualifiers.items():
        program = [("QueryRelationQualifier", [h, hc, r, tc, t, qkey])]
        program = get_program_seq(program)
        data.append((program, qv))
        
    # h t -> r
    program = [("QueryRelationBetween", [h, hc, tc, t])]
    program = get_program_seq(program)
    data.append((program, r))
    
    # h r t v -> q
    for qkey, (qc, qv) in qualifiers.items():
        program = [("QueryRelationQualifierName", [h, hc, r, tc, t, qv])]
        program = get_program_seq(program)
        data.append((program, qkey))
    
def add_attribute(hid, r, tid, qualifiers):
    hc, tc = get_concept(hid), get_concept(tid)
    h, t = get_name(hid), get_value(tid)[:200]
    for qkey, values in qualifiers.items():
        qualifiers[qkey] = (get_concept(values[0]), get_value(values[0]))
        
    # h r q v -> t
    program = [("QueryAttr", [h, hc, r])]
    for qkey, (qc, qv) in qualifiers.items():
        program.append(("FilterQualifier", [qkey, qv]))
    program = get_program_seq(program)
    data.append((program, t))
    
    # t r q v -> h
    program = [("FilterAttr", [t, r])]
    for qkey, (qc, qv) in qualifiers.items():
        program.append(("FilterQualifier", [qkey, qv]))
    program = get_program_seq(program)
    data.append((program, f'{hc} | {h}'))
    
    # h t r q -> v
    for qkey, (qc, qv) in qualifiers.items():
        program = [("QueryAttrQualifier", [h, hc, r, t, qkey])]
        program = get_program_seq(program)
        data.append((program, qv))
        
    # h t -> r
    program = [("QueryAttrName", [h, hc, t])]
    program = get_program_seq(program)
    data.append((program, r))
    
    # h r t v -> q
    for qkey, (qc, qv) in qualifiers.items():
        program = [("QueryAttrQualifierName", [h, hc, r, t, qv])]
        program = get_program_seq(program)
        data.append((program, qkey))

def add_instance(e, c):
    # e -> c
    program = [("InstanceOf", [e])]
    program = get_program_seq(program)
    data.append((program, c))
    
    # c -> e
    program = [("QueryInstance", [c])]
    program = get_program_seq(program)
    data.append((program, e))

def add_subclass(c, sc):
    # c -> sc
    program = [("SubclassOf", [c])]
    program = get_program_seq(program)
    data.append((program, sc))
    
    # sc -> c
    program = [("QuerySubClass", [sc])]
    program = get_program_seq(program)
    data.append((program, c))

r2cnt = {}
for r in tqdm(relation2facts):
    r2cnt[r] = len(relation2facts[r][:max_size])
    facts = sample_facts(relation2facts[r])
    for hid, _r, tid, qualifiers in facts:
        assert _r == r
        add_relation(hid, _r, tid, deepcopy(qualifiers))

for at in tqdm(attribute2facts):
    r2cnt[at] = len(attribute2facts[at][:max_size])
    facts = sample_facts(attribute2facts[at])
    for hid, _at, t, qualifiers in facts:
        assert _at == at
        add_attribute(hid, _at, t, deepcopy(qualifiers))

for cid in tqdm(cid2eids):
    eids = sample_facts(cid2eids[cid], repeat=True)
    for eid in eids:
        assert isinstance(eid, str)
        e = kb["entities"][eid]["name"]
        c = kb["concepts"][cid]['name']
        add_instance(e, c)
        
for cid, concept in tqdm(list(kb["concepts"].items())):
    c = concept["name"]
    for scid in concept["subclassOf"]:
        sc = kb["concepts"][scid]["name"]
        for _ in range(min_size):
            add_subclass(c, sc)

print(len(data))
json.dump([{"program":x, "answer":y} for (x, y) in data], open(f"{save_dir}/hrt.json", "w"), indent=2, ensure_ascii=False)