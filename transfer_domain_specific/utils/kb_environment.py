import random
import re
import time
import functools
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Union, Set

import json
from collections import Counter
from kopl.kopl_new import KoPLEngine
from .soay_api import *

count_keywords = ["how many", "number of", "amount of", "quantity of", "count of"]

def parse_seq_program(seq):
    chunks = seq.split('<func>')
    program = []
    for chunk in chunks:
        chunk = chunk.strip()
        res = chunk.split('<arg>')
        res = [_.strip() for _ in res]
        if len(res) > 0:
            func = res[0]
            inputs = []
            if len(res) > 1:
                for x in res[1:]:
                    inputs.append(x)
            else:
                inputs = []
            program.append({
                "function": func,
                "inputs": inputs,
            })
    return program

def find_entities(program):
    return [x["inputs"][0] for x in program if x["function"] == "Find" and len(x["inputs"]) > 0]

class Program:
    def __init__(
        self,
        source: Union[Set, str] = None,
        code: str = None,  
        function: str = None,
        branch_classes: list = None,
        execution: list = None,
        topic_entities: list = None,
    ):
        self.source = source
        self.code = code
        self.function = function
        self.execution = execution
        self.branch_classes = branch_classes
        self.topic_entities = topic_entities
        
        self.kopl = parse_seq_program(self.code)

    def __str__(self):
        return self.code_raw

class Computer:
    def __init__(self, dataset='soay'):
        self._dataset = dataset
        self.class_relations = defaultdict(set)
        self.class_attributes = defaultdict(set)
        self.r2domain = defaultdict(set)
        self.r2range = defaultdict(set)
        
        if dataset == 'soay':
            self.executor = KoPLEngine(json.load(open("../data/soaybench/kb.json")))
            self.soay_api = aminer_soay()
            r2domain_range = json.load(open("../data/soaybench/r2domain_range.json"))
        elif dataset == "metaqa":
            self.executor = KoPLEngine(json.load(open("../data/metaqa/kb.json")))
            r2domain_range = json.load(open("../data/metaqa/r2domain_range.json"))
            
        for r, item in r2domain_range.items():
            if item["type"] == "relation":
                for c in item["domain"]:
                    self.class_relations[c].add((r, 'forward'))
                    self.r2domain[r].add(c)
                for c in item["range"]:
                    if item["bidirection"]:
                        self.class_relations[c].add((r, "backward"))
                    self.r2range[r].add(c)
            else:
                for c in item["domain"]:
                    self.class_attributes[c].add(r)
                    self.r2domain[r].add(c)
                for c in item["range"]:
                    self.r2range[r].add(c)
    
    def execute_program(self, program, ignore_error=True, show_details=False):
        func_list, args_list = [x["function"] for x in program.kopl], [x["inputs"] for x in program.kopl]
        if self._dataset == 'soay':
            res = execute_with_soay_api(func_list, args_list, self.soay_api)
        else:
            res = self.executor.forward(func_list, args_list, ignore_error=ignore_error, show_details=show_details)
        return res if res is not None else ["Execution Error"]
    
    def get_classes_for_variable(self, v):
        classes = set()
        if v in self.executor.kb.name_to_id:
            for eid in self.executor.kb.name_to_id[v]:
                classes.update([self.executor.kb.entities[x]["name"] for x in self.executor.kb.entities[eid]['isA']])
        else:
            if self._dataset == "soay":
                return {"scholar", "organization", "field of research"}
            elif self._dataset == "metaqa":
                return {"tag"}
            else:
                raise NotImplementedError(v)
        return classes
    
    def get_initial_programs(self, topic_entities, question=None):
        initial_programs = []
        # topic_entities = [(v, self.get_classes_for_variable(v)) for v in topic_entities if v in self.executor.kb.name_to_id]
        topic_entities = [(v, self.get_classes_for_variable(v)) for v in topic_entities]
        for v, v_classes in topic_entities:
            if self._dataset == "metaqa" and "tag" in v_classes:
                code = f'FindAll <func> FilterStr <arg> topic <arg> {v}'
                end_classes = self.r2domain["topic"]
                initial_programs.append(Program(
                    source={v},
                    code=code,
                    function='FilterStr',
                    branch_classes=[end_classes],
                ))
            else:
                possible_relations = set()
                possible_attributes = set()
                for vc in v_classes:
                    possible_relations.update(self.class_relations[vc])
                    possible_attributes.update(self.class_attributes[vc])
                    
                for r, d in possible_relations:
                    code = f'Find <arg> {v} <func> Relate <arg> {r} <arg> {d}'
                    end_classes = self.r2range[r] if d == "forward" else self.r2domain[r]
                    initial_programs.append(Program(
                        source={v},
                        code=code,
                        function='Relate',
                        branch_classes=[end_classes],
                    ))
                    if self._dataset == "metaqa": continue
                    if self._dataset == 'soay' and 'scholar' not in end_classes: continue
                    for v2, v2_classes in topic_entities:
                        if v == v2: continue
                        inter_classes = v2_classes & end_classes
                        if len(inter_classes) > 0:
                            code2 = f'Find <arg> {v} <func> Relate <arg> {r} <arg> {d} <func> Find <arg> {v2} <func> And'
                            initial_programs.append(Program(
                                source={v, v2},
                                code=code2,
                                function="And",
                                branch_classes=[inter_classes],
                            ))
                                    
                for at in self.class_attributes[vc]:
                    code = f'Find <arg> {v} <func> QueryAttr <arg> {at}'
                    end_classes = self.r2range[at]
                    initial_programs.append(Program(
                            source={v},
                            code=code,
                            function='QueryAttr',
                            branch_classes=[end_classes],
                        ))
        return initial_programs
    
    def check_repeat(self, program, r):
        if len(program.kopl)==4 and [x["function"] for x in program.kopl] == ['Find', 'Relate', 'Find', 'And']:
            return r == program.kopl[1]["inputs"][0]
        return False
    
    def get_admissible_programs(self, program: Program,
                                initial_programs: Dict[Tuple, List[Program]],
                                question=None, expected_hop=None):
        candidate_programs = []
        branch_num = sum([(x["function"] in ['Find', 'FindAll']) for x in program.kopl])
        and_num = sum([(x["function"] == 'And') for x in program.kopl])
        
        # Relate / QueryAttr
        if program.function in ['FilterConcept', 'And']:
            possible_relations = set()
            possible_attributes = set()
            for c in program.branch_classes[-1]:
                possible_relations.update(self.class_relations[c])
                possible_attributes.update(self.class_attributes[c])
            for r, d in possible_relations:
                if self._dataset == "soay":
                    if d == "backward": continue
                    if self.check_repeat(program, r): continue
                code = f'{program.code} <func> Relate <arg> {r} <arg> {d}'
                end_classes = self.r2range[r] if d == "forward" else self.r2domain[r]
                candidate_programs.append(Program(
                    source=program.source,
                    code=code,
                    function='Relate',
                    branch_classes=program.branch_classes[:-1] + [end_classes]
                ))
            if branch_num - and_num == 1:
                for at in possible_attributes:
                    code = f'{program.code} <func> QueryAttr <arg> {at}'
                    end_classes = self.r2range[at]
                    candidate_programs.append(Program(
                        source=program.source,
                        code=code,
                        function='QueryAttr',
                        branch_classes=program.branch_classes[:-1] + [end_classes]
                    ))
        
        # FilterConcept
        if program.function in ['Relate', 'FilterStr']:
            for c in program.branch_classes[-1]:
                code = f'{program.code} <func> FilterConcept <arg> {c}'
                candidate_programs.append(Program(
                    source=program.source,
                    code=code,
                    function='FilterConcept',
                    branch_classes=program.branch_classes[:-1] + [{c}]
                ))
        
        # What
        if program.function in ['SelectAmong', 'FilterConcept', 'And'] and branch_num - and_num == 1:
            code = f'{program.code} <func> What'
            candidate_programs.append(Program(
                source=program.source,
                code=code,
                function='What',
                branch_classes=program.branch_classes
            ))
        
        if self._dataset == "metaqa":
            # print(expected_hop)
            res = []
            for p in candidate_programs:
                num_hop = sum([(x["function"] in ['Relate', 'QueryAttr', 'FilterStr']) for x in p.kopl])
                if num_hop > expected_hop:
                    continue
                elif num_hop < expected_hop:
                    if p.kopl[-1]["function"] in ["What", "QueryAttr"]:
                        continue 
                if num_hop >=3 and p.kopl[-1]["function"] == "Relate": 
                    relations = [x["inputs"][0] for x in p.kopl if x["function"] == "Relate"]
                    if len(relations) >= 3 and all((r == relations[0]) for r in relations):
                        continue  
                res.append(p)
            return res
        
        # SelectAmong
        if program.function in ['FilterConcept']:
            possible_attributes = set()
            for c in program.branch_classes[-1]:
                possible_attributes.update(self.class_attributes[c])
            for at in possible_attributes:
                if "quantity" not in self.r2range[at]: continue
                for comp in ['largest', 'smallest']:
                    code = f'{program.code} <func> SelectAmong <arg> {at} <arg> {comp}'
                    candidate_programs.append(Program(
                        source=program.source,
                        code=code,
                        function='SelectAmong',
                        branch_classes=program.branch_classes
                    ))
        
        # add a new branch
        def no_intersect_sources(source1, source2):
            if self._dataset == "soay":
                for v1 in source1:
                    v1 = v1.lower()
                    if v1.startswith("the "):
                        v1 = v1[4:].strip()
                    for v2 in source2:
                        v2 = v2.lower()
                        if v2.startswith("the "):
                            v2 = v1[4:].strip()
                        if v1 in v2 or v2 in v1:
                            return False
                return True
            else:
                return len(set(source2) & set(source1)) == 0
            
        if len(initial_programs) > 1 and program.function in ['FilterConcept', 'Relate']:
            for source2 in initial_programs:
                if no_intersect_sources(program.source, source2):
                    for program2 in initial_programs[source2]:
                        if program2.function in ["QueryAttr"]: continue
                        code = f'{program.code} <func> {program2.code}'
                        candidate_programs.append(Program(
                            source=tuple(list(program.source) + list(program2.source)),
                            code=code,
                            function=program2.function,
                            branch_classes=program.branch_classes + program2.branch_classes
                        )) 
                else:
                    for program2 in initial_programs[source2]:
                        if program2.code != program.code and program2.code.startswith(program.code):
                            candidate_programs.append(program2)
        
        # And
        if program.function in ['FilterConcept'] and branch_num > 1 and branch_num - and_num > 1:
            inter_classes = program.branch_classes[-1] & program.branch_classes[-2]
            if len(inter_classes) > 0:
                code = f'{program.code} <func> And'
                candidate_programs.append(Program(
                    source=program.source,
                    code=code,
                    function='And',
                    branch_classes=program.branch_classes[:-2] + [inter_classes]
                ))
        
        # count
        # if program.function in ['FilterConcept', 'And'] and branch_num - and_num == 1 and any(x in question.lower() for x in count_keywords):
        #     code = f'{program.code} <func> Count'
        #     candidate_programs.append(Program(source=program.source,
        #                                             code=code,
        #                                             function='Count',
        #                                             branch_classes=program.branch_classes[:-1] + ["quantity"]
        #                                             ))
        
        return candidate_programs

def execute_with_soay_api(func_list, args_list, api):
    branches = []
    i = 0

    # SearchPerson
    while i < len(func_list):
        if func_list[i:i+4] == ['Find', 'Relate', 'Find', 'And']:
            kwargs = {}
            kwargs["name"] = args_list[i+2][0]
            result = api.searchPersonComp(**kwargs)
            if len(result) > 1:
                if args_list[i+1][0] == 'organization':
                    kwargs["organization"] = args_list[i][0]
                elif args_list[i+1][0] == 'field of research':
                    kwargs["interest"] = args_list[i][0]
                result = api.searchPersonComp(**kwargs)
            branches.append([{
                "func": "searchPersonComp",
                "result": result,
                "return": [x["name"] for x in result],
            }])
            i += 4
            
        elif func_list[i] == 'Find':
            if i < len(func_list) and args_list[i+1] == ['organization', 'backward']:
                if len(branches) > 0 and branches[-1][-1]["func"] == "getCoauthors" and func_list[i:i+4] == ["Find", "Relate", "FilterConcept", "And"]:
                    result = []
                    for x in branches[-1][-1]["result"]:
                        try:
                            info = api.searchPersonComp(name=x["name"])
                            y = info[0]
                            if y["person_id"]==x["person_id"] and y["organization"] == args_list[i][0]:
                                result.append(x)
                        except:
                            info = api.getPersonBasicInfo(person_id=x["person_id"])
                            if info["organization"] == args_list[i][0]:
                                result.append(x)
                        # info = api.searchPersonComp(name=x["name"])
                        # y = info[0]
                        # if y["organization"] == args_list[i][0]:
                        #     result.append(x)
                    branches[-1].append({
                        "func": "getCoauthors",
                        "result": result,
                        "return": [x["name"] for x in result]
                    })
                    i += 4
                else:
                    result = api.searchPersonComp(organization=args_list[i][0])
                    branches.append([{
                        "func": "searchPersonComp",
                        "result": result,
                        "return": [x["name"] for x in result],
                    }])
                    i += 2
            elif i < len(func_list) and args_list[i+1] == ['field of research', 'backward']:
                if len(branches) > 0 and branches[-1][-1]["func"] == "getCoauthors" and func_list[i:i+4] == ["Find", "Relate", "FilterConcept", "And"]:
                    result = []
                    for x in branches[-1][-1]["result"]:
                        # print("yes")
                        # try:
                        #     info = api.searchPersonComp(name=x["name"])
                        #     y = info[0]
                        #     if y["person_id"]==x["person_id"] and args_list[i][0] in y["interests"]:
                        #         result.append(x)
                        # except:
                        #     interests = api.getPersonInterest(person_id=x["person_id"])
                        #     if args_list[i][0] in interests:
                        #         result.append(x)
                        interests = api.getPersonInterest(person_id=x["person_id"])
                        if args_list[i][0] in interests:
                            result.append(x)
                    branches[-1].append({
                        "func": "getCoauthors",
                        "result": result,
                        "return": [x["name"] for x in result]
                    })
                    i += 4
                elif len(branches) == 0 and func_list[i:i+7] == ["Find", "Relate", "FilterConcept", "Find", "Relate", "FilterConcept", "And"] and args_list[i+4] == ["collaborator", "forward"]:
                    result = api.searchPersonComp(interest=args_list[i][0])
                    result2 = []
                    for x in result:
                        p_id = api.searchPersonComp(name=x["name"])[0]['person_id']
                        coauthors = [y["name"] for y in api.getCoauthors(p_id)]
                        if args_list[i+3][0] in coauthors:
                            result2.append(x)
                    branches.append([{
                        "func": "searchPersonComp",
                        "result": result2,
                        "return": [x["name"] for x in result2],
                    }])
                    i += 7
                else:
                    result = api.searchPersonComp(interest=args_list[i][0])
                    branches.append([{
                        "func": "searchPersonComp",
                        "result": result,
                        "return": [x["name"] for x in result],
                    }])
                    i += 2
            else: 
                result = api.searchPersonComp(name=args_list[i][0])
                branches.append([{
                    "func": "searchPersonComp",
                    "result": result,
                    "return": [x["name"] for x in result],
                }])
                i += 1
                
        elif func_list[i] == 'Relate':
            r, d = args_list[i]
            assert d == 'forward'
            # SearchPerson
            if r in ['field of research', 'organization']:
                func, result = branches[-1][-1]["func"], branches[-1][-1]["result"]
                assert func == "searchPersonComp"
                branches[-1].append({
                    "func": func,
                    "result": result,
                    "return": [result[0]["organization"]] if r == 'organization' else result[0]["interests"],
                })
            # getCoauthor
            elif r in ['collaborator']:
                func, result = branches[-1][-1]["func"], branches[-1][-1]["result"]
                if func == "searchPersonComp":
                    new_result = api.getCoauthors(result[0]['person_id'])
                    branches[-1].append({
                        "func": "getCoauthors",
                        "result": new_result,
                        "return": [x["name"] for x in new_result],
                    })
                elif func == "getCoauthors":
                    new_result = []
                    for coauthor in result:
                        new_result.extend(api.getCoauthors(coauthor['person_id']))
                    branches[-1].append({
                        "func": "getCoauthors",
                        "result": new_result,
                        "return": list(set([x["name"] for x in new_result])),
                    })
                else:
                    raise NotImplementedError
            # getPersonPubs
            elif r in ['publications', 'representative work']:
                func, result = branches[-1][-1]["func"], branches[-1][-1]["result"]
                assert func == "searchPersonComp"
                new_result = api.getPersonPubs(result[0]['person_id'])
                if r == 'representative work':
                    new_result = new_result[:1]
                branches[-1].append({
                    "func": "getPersonPubs",
                    "result": new_result,
                    "return": [x['title'] for x in new_result], 
                })
            elif r in ['author']:
                func, result = branches[-1][-1]["func"], branches[-1][-1]["result"]
                assert func == "getPersonPubs"
                branches[-1].append({
                    "func": "getPersonPubs",
                    "result": result,
                    "return": result[0]['authors_name_list'], 
                })
            elif r in ["published in"]:
                func, result = branches[-1][-1]["func"], branches[-1][-1]["result"]
                assert func == "getPersonPubs"
                new_result = api.getPublication(result[0]['pub_id'])
                branches[-1].append({
                    "func": "getPublication",
                    "result": new_result,
                    "return": [new_result["venue"]['name'] if 'name' in new_result["venue"] else new_result["venue"]["info"]["name"]], 
                })
            i += 1
            
        elif func_list[i] == 'QueryAttr':
            at = args_list[i][0]
            
            person_basic_info_key = {
                "gender": "gender",
                "introduction": "bio",
                "educational background": "education_experience",
                "title": "position",
                "email": "email",
            }
            pub_info_key = {
                "abstract": "abstract", 
                "pdf link": "pdf_link",
            }
            
            # SearchPserson, getPersonPub
            if at in ['citation count', 'number of publications']:
                func, result = branches[-1][-1]["func"], branches[-1][-1]["result"]
                assert func in ["searchPersonComp", "getPersonPubs"]
                branches[-1].append({
                    "func": func,
                    "result": result,
                    "return": [result[0]["num_citation"] if at == 'citation count' else result[0]["num_pubs"]],
                })
            elif at in ["year of publication"]:
                func, result = branches[-1][-1]["func"], branches[-1][-1]["result"]
                assert func == "getPersonPubs"
                branches[-1].append({
                    "func": func,
                    "result": result,
                    "return": [result[0]["year"]],
                })
            # getPersonBasicInfo
            elif at in person_basic_info_key:
                func, result = branches[-1][-1]["func"], branches[-1][-1]["result"]
                assert func == "searchPersonComp"
                new_result = api.getPersonBasicInfo(result[0]['person_id'])
                branches[-1].append({
                    "func": "getPersonBasicInfo",
                    "result": new_result,
                    "return": [new_result[person_basic_info_key[at]]], 
                })
            elif at in pub_info_key:
                func, result = branches[-1][-1]["func"], branches[-1][-1]["result"]
                assert func == "getPersonPubs"
                new_result = api.getPublication(result[0]['pub_id'])
                branches[-1].append({
                    "func": "getPublication",
                    "result": new_result,
                    "return": [new_result[pub_info_key[at]]], 
                })
            elif at in ["published in"]:
                func, result = branches[-1][-1]["func"], branches[-1][-1]["result"]
                assert func == "getPersonPubs"
                new_result = api.getPublication(result[0]['pub_id'])
                branches[-1].append({
                    "func": "getPublication",
                    "result": new_result,
                    "return": [new_result["venue"]['name'] if 'name' in new_result["venue"] else new_result["venue"]["info"]["name"]], 
                })
            i += 1
        elif func_list[i] == "And":
            assert len(branches) > 1
            id2name = {x["person_id"]: x["name"] for x in branches[-2][-1]["result"]}
            func, result = branches[-1][-1]["func"], branches[-1][-1]["result"]
            new_result = [x for x in result if x["person_id"] in id2name]
            branches[-1].append({
                "func": func,
                "result": new_result,
                "return": [x["name"] for x in new_result], 
            })
            i += 1
        elif func_list[i] == "SelectAmong":
            at, comp = args_list[i]
            func, result = branches[-1][-1]["func"], branches[-1][-1]["result"]
            assert func in ["getCoauthors", "searchPersonComp"] and at in ['number of publications', 'citation count'], f"{func}\t{at}"
            new_result = []
            id2name = {}
            for x in result:
                new_result.extend([y for y in api.searchPersonComp(name=x["name"]) if y["person_id"] == x["person_id"]])
            new_result = sorted(new_result, reverse=comp=="largest", key=lambda x:(x["num_citation"] if at == 'citation count' else x["num_pubs"]))[:1]
            branches[-1].append({
                "func": func,
                "result": new_result,
                "return": [x["name"] for x in new_result], 
            })
            i += 1
        elif func_list[i] in ["FilterConcept", "What"]:
            i += 1
        else:
            raise NotImplementedError
    return [str(x) for x in branches[-1][-1]["return"]]
            
            
            
                
            
                
        
                
            