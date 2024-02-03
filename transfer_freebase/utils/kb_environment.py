import random
import re
import time
import functools
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Union, Set

from .logic_form_util import kopl_to_sparql, parse_seq_program
from .sparql_cache import SparqlCache
import json
from collections import Counter

data_type2filter_func = {
    "int": "FilterNum",
    "integer": "FilterNum",
    "float": "FilterNum",
    "double": "FilterNum",
    "gYear": "FilterYear",
    "date": "FilterDate",
    "dateTime": "FilterDate",
    "gYearMonth": "FilterDate",
}

count_keywords = ["how many", "number of", "amount of", "quantity of", "count of"]
webq_tc_now = {
    "people.marriage.spouse": "people.marriage.from",
    "sports.sports_team_roster.team": "sports.sports_team_roster.from",
    "government.government_position_held.office_holder": "government.government_position_held.from"
}


def get_last_branch(code:str):
    chunks = code.split(' <func> ')
    idx = 0
    for i, chunk in enumerate(chunks):
        if chunk.startswith('Find'):
            idx = i
    return ' <func> '.join(chunks[:idx]), ' <func> '.join(chunks[idx:])

class Program:
    def __init__(
        self,
        source: Union[Set, str] = None,
        code: str = None,  # used for PLM classifier
        code_raw: str = None,  # original code (i.e., code with mids)
        function: str = None,
    ):
        """
        :param source: anchor entities/literals
        :param code: programs with readable entity names
        :param code_raw: original programs
        :param function: function name of the outmost subprogram
        :param height: height
        :param execution: execution results or an arg class
        :param finalized: whether it is a finalized program
        :param derivations: relations paths (optionally with comparators) indexed by different source nodes
        """
        self.source = source
        self.code = code
        self.code_raw = code_raw
        self.function = function
        self.execution = None
        self.kopl = parse_seq_program(self.code_raw)

    def __str__(self):
        return self.code_raw

path = str(Path(__file__).parent.absolute())


def get_vocab(dataset: str):
    if dataset == "grail":
        with open(path + '/vocab_files/grailqa.json') as f:
            data = json.load(f)
        return set(data["relations"]), set(data["classes"]), set(data["attributes"])
    elif dataset == "gq1":
        with open(path + '/vocab_files/gq1.json') as f:
            data = json.load(f)
        return set(data["relations"]), set(data["classes"]), set(data["attributes"])
    elif dataset == "webq":
        with open(path + '/vocab_files/gq1.json') as f:
            data = json.load(f)
        classes = set(data["classes"])
        with open(path + '/vocab_files/webq_correct.json') as f:
            data = json.load(f)
        return set(data["relations"]), classes, set(data["attributes"]), set(data["tc_attributes"]), set(
            data["cons_attributes"]), data["cons_ids"], set(data["classes"])
    elif dataset == "cwq":
        pass

def load_ambiguous_relations(dataset):
    r2cnt = {}
    if dataset == "grail":
        return None
    else:
        with open(path + f'/vocab_files/{dataset}_disambiguities.json') as f:
            data = json.load(f)
    for name in data["relations"]:
        r2cnt.update({x:1/(i+1) for i, x in enumerate(data["relations"][name])})
    for name in data["attributes"]:
        r2cnt.update({x:1/(i+1) for i, x in enumerate(data["attributes"][name])})
    return r2cnt

def get_name(x):
    x = x.split('.')[-1]
    if x.endswith('_s'):
        x = x[:-2]
    x = x.replace('_s_', '')
    x = ' '.join(x.split('_'))
    return x

def merge_relation(r1, r2):
    return r2

def get_ontology(dataset: str):
    class_hierarchy = defaultdict(lambda: [])
    class_out_edges = defaultdict(lambda: set())
    class_in_edges = defaultdict(lambda: set())
    relation_domain = {}
    relation_range = {}
    date_attributes = set()
    numerical_attributes = set()
    if dataset == "grail":
        fb_type_file = path + "/../ontology/commons/fb_types"
        fb_roles_file = path + "/../ontology/commons/fb_roles"
    elif dataset == "gq1":
        fb_type_file = path + "/../ontology/fb_types"
        fb_roles_file = path + "/../ontology/fb_roles"

    else:  # webq does not need these information
        return class_out_edges, class_in_edges, relation_domain, relation_range, date_attributes, numerical_attributes

    with open(fb_type_file) as f:
        for line in f:
            fields = line.split()
            if fields[2] != "common.topic":
                class_hierarchy[fields[0]].append(fields[2])

    with open(fb_roles_file) as f:
        for line in f:
            fields = line.split()
            relation_domain[fields[1]] = fields[0]
            relation_range[fields[1]] = fields[2]

            class_out_edges[fields[0]].add(fields[1])
            class_in_edges[fields[2]].add(fields[1])

            if fields[2] in ['type.int', 'type.float']:
                numerical_attributes.add(fields[1])
            elif fields[2] == 'type.datetime':
                date_attributes.add(fields[1])

    for c in class_hierarchy:
        for c_p in class_hierarchy[c]:
            class_out_edges[c].update(class_out_edges[c_p])
            class_in_edges[c].update(class_in_edges[c_p])

    return class_out_edges, class_in_edges, relation_domain, relation_range, date_attributes, numerical_attributes


class Computer:
    def __init__(self, dataset='grail'):
        self._dataset = dataset
        if dataset in ["grail", "gq1"]:
            self._relations, self._classes, self._attributes = get_vocab(dataset)
        elif dataset == "webq":
            self._relations, self._classes, self._attributes, self._tc_attributes, self._cons_attributes, self._cons_ids, self._classes_vocab = get_vocab(
                dataset)
        
        if dataset == "grail":
            with open('ontology/domain_dict', 'r') as f:
                self._domain_dict = json.load(f)
            with open('ontology/domain_info', 'r') as f:
                self._domain_info = json.load(f)
        self._class_out, self._class_in, self._relation_d, self._relation_r, self._date_attributes, \
        self._numerical_attributes = get_ontology(dataset)
        self._date_attributes = self._date_attributes.intersection(self._attributes)
        self._numerical_attributes = self._numerical_attributes.intersection(self._attributes)
        self._cache = SparqlCache(dataset)

        self.cvt_types = set()
        with open(path + "/../ontology/cvt_types.txt") as f:
            for line in f:
                self.cvt_types.add(line.replace('\n', ''))
        self.cvt_relations = set()
        with open(path + "/../ontology/cvt_relations.txt") as f:
            for line in f:
                self.cvt_relations.add(line.strip())
        
        self.ambiguous_relation_score = load_ambiguous_relations(dataset)
    
    def choose_final_program(self, programs, k=1):
        if len(programs) == 1 or self.ambiguous_relation_score is None:
            return programs[0] if k == 1 else programs
        # return programs[0]
        name2r = defaultdict(dict)
        for i, program in enumerate(programs):
            # print(program.code_raw)
            for x in program.kopl:
                for arg in x["inputs"]:
                    if arg in self.ambiguous_relation_score:
                        if i not in name2r[get_name(arg)]:
                            name2r[get_name(arg)][i] = (arg, self.ambiguous_relation_score[arg])
                        elif self.ambiguous_relation_score[arg] > name2r[get_name(arg)][i][1]:
                            name2r[get_name(arg)][i] = (arg, self.ambiguous_relation_score[arg])
                              
        scores = {i:0 for i in range(len(programs))}
        # print(name2r)
        for name in name2r:
            if len(name2r[name]) == len(programs):
                program_scores = [(idx, x[1]) for idx, x in name2r[name].items()]
                for i, (idx, _) in enumerate(sorted(program_scores, reverse=True, key=lambda x:x[1])):
                    scores[idx] += 1.0 / (i+1)
        if k == 1:
            max_score = max(scores.values())
            for i in scores:
                if scores[i] == max_score:
                    return programs[i]
        else:
            return [programs[idx] for (idx, score) in sorted(list(scores.items()), reverse=True, key=lambda x:x[1])][:k]
                    
    def get_vocab(self):
        return self._relations, self._classes, self._attributes

    def process_value(self, value):
        data_type = value.split("^^")[1].split("#")[1]
        if data_type not in ['integer', 'float', 'double', 'dateTime']:
            value = f'"{value.split("^^")[0] + "-08:00"}"^^<{value.split("^^")[1]}>'
            # value = value.split("^^")[0] + '-08:00^^' + value.split("^^")[1]
        else:
            value = f'"{value.split("^^")[0]}"^^<{value.split("^^")[1]}>'

        return value

    def get_relations_for_program(self, program, reverse=False):
        # print(program.code_raw)
        sparql_query, entities = kopl_to_sparql(program.kopl, return_entities=True)
        clauses = sparql_query.split("\n")
        if reverse:
            new_clauses = [clauses[0], "SELECT DISTINCT ?rel\nWHERE {\n?y ?rel ?x .\n{"]
        else:
            new_clauses = [clauses[0], "SELECT DISTINCT ?rel\nWHERE {\n?x ?rel ?y .\n{"]
        new_clauses.extend(clauses[1:])
        new_clauses.append("}")
        for entity in entities:
            new_clauses.append('FILTER (?y != :%s)' % entity)
        new_clauses.append("}")

        new_query = '\n'.join(new_clauses)
        results = self.execute_SPARQL(new_query)

        rtn = results.intersection(self._relations)

        return sorted(rtn)

    def get_relations_for_variables(self, entities, reverse=False):
        rtn = set()
        for entity in list(entities)[:100]:
            try:
                if reverse:
                    rtn.update(self._cache.get_in_relations(entity).intersection(self._relations))
                else:
                    rtn.update(self._cache.get_out_relations(entity).intersection(self._relations))
            except Exception:
                # print("entity:", entity)
                pass
            
        return sorted(rtn)

    def get_attributes_for_program(self, program):
        sparql_query = kopl_to_sparql(program.kopl)
        clauses = sparql_query.split("\n")
        new_clauses = [clauses[0], "SELECT DISTINCT ?att\nWHERE {\n?x ?att ?obj .\n{"]
        new_clauses.extend(clauses[1:])
        new_clauses.append("}\n}")

        new_query = '\n'.join(new_clauses)
        results = self.execute_SPARQL(new_query)

        rtn = results.intersection(self._attributes)

        return sorted(rtn)

    def get_attributes_for_variables(self, entities):
        rtn = set()
        for entity in list(entities)[:100]:
            try:
                rtn.update(self._cache.get_out_relations(entity).intersection(self._attributes))
            except Exception:
                # print("entity:", entity)
                pass

        return sorted(rtn)

    def get_attributes_for_value(self, value, use_ontology=False):
        rtn = set()

        if use_ontology:
            if value.__contains__("#float") or value.__contains__("#integer") or value.__contains__("#double"):
                rtn.update(self._numerical_attributes)
            else:
                rtn.update(self._date_attributes)
        else:  # retrieve based on KB facts
            data_type = value.split("#")[1]
            if data_type not in ['integer', 'float', 'double', 'dateTime']:
                value = f'"{value.split("^^")[0] + "-08:00"}"^^<{value.split("^^")[1]}>'
            else:
                value = f'"{value.split("^^")[0]}"^^<{value.split("^^")[1]}>'

            rtn.update(self._cache.get_in_attributes(value).intersection(self._attributes))

        return sorted(rtn)

    def get_attributes_for_class(self, class_name):
        return sorted(self._class_out[class_name].intersection(self._attributes))

    def get_constraints_for_program(self, program):
        sparql_query = kopl_to_sparql(program.kopl)
        clauses = sparql_query.split("\n")
        new_clauses = [clauses[0], "SELECT DISTINCT ?att\nWHERE {\n?x ?att ?obj .\n{"]
        new_clauses.extend(clauses[1:])
        new_clauses.append("}\n}")

        new_query = '\n'.join(new_clauses)
        results = self.execute_SPARQL(new_query)

        rtn = results.intersection(self._cons_attributes)
        return sorted(rtn)
    
    def get_tc_constraints_for_program(self, program):
        sparql_query = kopl_to_sparql(program.kopl)
        clauses = sparql_query.split("\n")
        new_clauses = [clauses[0], "SELECT DISTINCT ?att\nWHERE {\n?x ?att ?obj .\n{"]
        new_clauses.extend(clauses[1:])
        new_clauses.append("}\n}")

        new_query = '\n'.join(new_clauses)
        results = self.execute_SPARQL(new_query)

        rtn = results.intersection(self._tc_attributes)
        return sorted(rtn)

    def is_intersectant(self, derivation1, derivation2):
        return self._cache.is_intersectant(derivation1, derivation2)
    
    def get_intersection(self, program_0, program_1):
        sparql_0 = kopl_to_sparql(program_0.kopl)
        result_0 = self.execute_SPARQL(sparql_0)
        sparql_1 = kopl_to_sparql(program_1.kopl)
        result_1 = self.execute_SPARQL(sparql_1)
        return result_0 & result_1

    def get_classes_for_program(self, program):
        sparql_query = kopl_to_sparql(program.kopl)
        clauses = sparql_query.split("\n")
        new_clauses = [clauses[0], "SELECT DISTINCT ?cls\nWHERE {\n?x :type.object.type ?cls .\n{"]
        new_clauses.extend(clauses[1:])
        new_clauses.append("}\n}")

        new_query = '\n'.join(new_clauses)
        results = self.execute_SPARQL(new_query)
        rtn = results.intersection(self._classes)

        return sorted(rtn)

    def execute_program(self, program):
        sparql_query = kopl_to_sparql(program.kopl)
        rtn = self.execute_SPARQL(sparql_query)
        return rtn

    def execute_SPARQL(self, sparql_query):
        rtn = self._cache.get_sparql_execution(sparql_query)
        return set(rtn)
    
    def check_cvt(self, program):
        possible_types = self.get_classes_for_program(program)
        if len(possible_types) == 0:
            return False
        
        for t in possible_types:
            if t not in self.cvt_types:
                return False
        return True
    
    def is_query_attr(self, question):
        if question.startswith("what year") or question.startswith("when ") or question.startswith("what age "):
            return True
        if any(x in question for x in [" name", " called", " code", " nickname"]):
            return True
        if question.startswith("how ") and not question.startswith("how many"):
            return True
        return False
    
    def check_time(self, question):
        if " first " in question:
            return "earliest"
        for word in [" did ", " was ", " were ", " has "]:
            if word in question:
                return "past"
        return "now"

    def get_admissible_programs(self, program: Program,
                                initial_programs: Dict[str, List[Program]],
                                entity_name=None, question=None):

        candidate_programs = []
        
        branch_num = sum([(x["function"] in ['Find', 'FindAll']) for x in program.kopl])
        and_num = sum([(x["function"] == 'And') for x in program.kopl])
            
        # Relate
        if self._dataset != "webq" and program.function in ['FilterConcept', 'SelectAmong', 'And']:
            possible_relations = [(r, 'forward') for r in self.get_relations_for_program(program)]
            possible_relations.extend([(r, 'backward') for r in self.get_relations_for_program(program, reverse=True)])
            for r, d in possible_relations:
                code = f'{program.code} <func> Relate <arg> {get_name(r)} <arg> {d}'
                code_raw = f'{program.code_raw} <func> Relate <arg> {r} <arg> {d}'
                new_program = Program(source=program.source,
                                    code=code,
                                    code_raw=code_raw,
                                    function='Relate',
                                )
                if True:
                    candidate_programs.append(new_program)
        # print(len(candidate_programs))

        # SelectAmong
        if self._dataset != "webq" and program.function == 'FilterConcept':
            possible_attributes = self.get_attributes_for_program(program)
            for a in possible_attributes:
                for comp in ['largest', 'smallest']:
                    code = f'{program.code} <func> SelectAmong <arg> {get_name(a)} <arg> {comp}'
                    code_raw = f'{program.code_raw} <func> SelectAmong <arg> {a} <arg> {comp}'
                    candidate_programs.append(Program(source=program.source,
                                                        code=code,
                                                        code_raw=code_raw,
                                                        function='SelectAmong',
                                                        ))
        # print(len(candidate_programs))

        # FilterConcept
        if program.function in ['Relate', 'FilterNum', 'FilterDate', 'FilterYear']:
            possible_types = self.get_classes_for_program(program)
            for t in possible_types:
                code = f'{program.code} <func> FilterConcept <arg> {get_name(t)}'
                code_raw = f'{program.code_raw} <func> FilterConcept <arg> {t}'
                candidate_programs.append(Program(source=program.source,
                                                    code=code,
                                                    code_raw=code_raw,
                                                    function='FilterConcept',
                                                    ))
        # print(len(candidate_programs))
        
        # And
        if program.function in ['FilterConcept', 'SelectAmong']:
            if branch_num > 1 and branch_num - and_num > 1:
                code_raw_0, code_raw_1 = get_last_branch(program.code_raw)
                program_0, program_1 = Program(code_raw=code_raw_0), Program(code_raw=code_raw_1)
                intersection = self.get_intersection(program_0, program_1)
                if len(intersection) > 0:
                    code = f'{program.code} <func> And'
                    code_raw = f'{program.code_raw} <func> And'
                    candidate_programs.append(Program(source=program.source,
                                                        code=code,
                                                        code_raw=code_raw,
                                                        function='And',
                                                        ))
        # print(len(candidate_programs))
        
        # add a new branch
        if len(initial_programs) > 1 and program.function in ['FilterConcept', 'Relate'] and isinstance(program.source, str):
            for source in initial_programs:
                if source != program.source and isinstance(source, str) and ((program.source in entity_name) or (source in entity_name)):
                    for new_program in initial_programs[source]:
                        if new_program.function in ["QueryAttr", "QueryRelationQualifier"]: continue
                        code = f'{program.code} <func> {new_program.code}'
                        code_raw = f'{program.code_raw} <func> {new_program.code_raw}'
                        candidate_programs.append(Program(source={program.source, new_program.source},
                                                            code=code,
                                                            code_raw=code_raw,
                                                            function=new_program.function,
                                                            ))
        
        # add unended program
        for source2 in initial_programs:
            for program2 in initial_programs[source2]:
                if program2.code != program.code and program2.code.startswith(program.code):
                    candidate_programs.append(program2)
        # print(len(candidate_programs))
        
        if self.check_cvt(program):
            return candidate_programs
        
        # What
        if program.function in ['SelectAmong', 'FilterConcept', 'And'] and branch_num - and_num == 1:
            code = f'{program.code} <func> What'
            code_raw = f'{program.code_raw} <func> What'
            candidate_programs.append(Program(source=program.source,
                                                    code=code,
                                                    code_raw=code_raw,
                                                    function='What',
                                                    ))
        # print(len(candidate_programs))
        
        # Count
        if self._dataset != "webq" and program.function in ['FilterConcept', 'And'] and branch_num - and_num == 1 and any(x in question for x in count_keywords):
            code = f'{program.code} <func> Count'
            code_raw = f'{program.code_raw} <func> Count'
            candidate_programs.append(Program(source=program.source,
                                                    code=code,
                                                    code_raw=code_raw,
                                                    function='Count',
                                                    ))
        # print(len(candidate_programs))
                        
        return candidate_programs

    # @timer
    def get_initial_programs(self, entity_name, answer_types, question):
    
        initial_programs = []
        for v in entity_name:
            if v[:2] in ['m.', 'g.']:
                other_entities = {x for x in entity_name if (x != v and x[:2] in ['m.', 'g.'])}
                possible_years = {x for x in entity_name if re.match("[\d]{4}", x)}
                possible_relations = [(r, 'forward') for r in self.get_relations_for_variables({v})]
                possible_relations.extend((r, 'backward') for r in self.get_relations_for_variables({v}, reverse=True))
                for r, d in possible_relations:
                    code = f'Find <arg> {entity_name[v]} <func> Relate <arg> {get_name(r)} <arg> {d}'
                    code_raw = f'Find <arg> {v} <func> Relate <arg> {r} <arg> {d}'
                    program = Program(source=v,
                                    code=code,
                                    code_raw=code_raw,
                                    function='Relate')
                    if r in self.cvt_relations and self.check_cvt(program):
                        
                        possible_r2 = self.get_relations_for_program(program, reverse=(d=='backward'))
                        possible_at2 = self.get_attributes_for_program(program)
                        for r2 in possible_r2:
                            # Relate
                            code = f'Find <arg> {entity_name[v]} <func> Relate <arg> {merge_relation(get_name(r), get_name(r2))} <arg> {d}'
                            # add NOW time constrain for webq
                            if self._dataset == "webq" and d == "forward" and r2 in webq_tc_now and len(possible_years) == 0:
                                tq = webq_tc_now[r2]
                                time = self.check_time(question)
                                if time == "earliest":
                                    code_raw = f'Find <arg> {v} <func> Relate <arg> {r} <arg> {d} <func> SelectAmong <arg> {tq} <arg> smallest <func> Relate <arg> {r2} <arg> {d}'
                                elif time == "past":
                                    code_raw = f'Find <arg> {v} <func> Relate <arg> {r} <arg> {d} <func> Relate <arg> {r2} <arg> {d}'
                                elif time == "now":
                                    code_raw = f'Find <arg> {v} <func> Relate <arg> {r} <arg> {d} <func> ConstrainYear <arg> {tq} <arg> NOW <func> Relate <arg> {r2} <arg> {d}'
                                else:
                                    raise NotImplementedError(question)
                                if len(self.execute_program(Program(code_raw=code_raw))) == 0:
                                    code_raw = f'Find <arg> {v} <func> Relate <arg> {r} <arg> {d} <func> Relate <arg> {r2} <arg> {d}'
                            else:
                                code_raw = f'Find <arg> {v} <func> Relate <arg> {r} <arg> {d} <func> Relate <arg> {r2} <arg> {d}'
                                
                            program2 = Program(source=v,
                                              code=code,
                                              code_raw=code_raw,
                                              function='Relate')
                            initial_programs.append(program2)
                        
                            # QueryRelationQualifier
                            if len(other_entities) > 0 and d == 'forward':
                                possible_v2 = list(self.execute_program(program2) & other_entities)
                                if len(possible_v2) > 0:
                                    possible_q = [x for x in possible_r2 if x != r2]
                                    if self._dataset == "webq" and self.is_query_attr(question):
                                        possible_q = possible_q + [x for x in possible_at2 if ".from" in x]
                                    # print(possible_q)
                                    for v2 in possible_v2:
                                        for q in possible_q:
                                            code = f'Find <arg> {entity_name[v]} <func> Find <arg> {entity_name[v2]} <func> QueryRelationQualifier <arg> {merge_relation(get_name(r), get_name(r2))} <arg> {merge_relation(get_name(r), get_name(q))}'
                                            code_raw = f'Find <arg> {v} <func> Relate <arg> {r} <arg> forward <func> Find <arg> {v2} <func> Relate <arg> {r2} <arg> backward <func> And <func> Relate <arg> {q} <arg> forward <func> What'
                                            initial_programs.append(Program(source={v, v2}, 
                                                               code=code,
                                                               code_raw=code_raw,
                                                               function='QueryRelationQualifier'))   
                        
                            # QFilter - constrain and time constrain
                            if self._dataset == "webq":
                                cons_entities = {x for x in entity_name if x != v and (x[:2] in ['m.', 'g.'] or (len(x) <= 2 and re.match('[\d]{1}', x)))}
                                if len(cons_entities) > 0 and d == 'forward':
                                    # constrain
                                    possible_qs = [q for q in self.get_constraints_for_program(program) if q != r2]
                                    for q in possible_qs:
                                        program_q = Program(
                                            code_raw=f'Find <arg> {v} <func> Relate <arg> {r} <arg> forward <func> Relate <arg> {q} <arg> forward'
                                        )
                                        possible_qvs = list(self.execute_program(program_q) & cons_entities)
                                        if len(possible_qvs) > 0:
                                            for qv in possible_qvs:
                                                if qv[:2] in ['m.', 'g.']:
                                                    code = f'Find <arg> {entity_name[v]} <func> Relate <arg> {merge_relation(get_name(r), get_name(r2))} <arg> forward <func> QFilterStr <arg> {merge_relation(get_name(r), get_name(q))} <arg> {entity_name[qv]}'
                                                    code_raw = f'Find <arg> {v} <func> Relate <arg> {r} <arg> forward <func> Find <arg> {qv} <func> Relate <arg> {q} <arg> backward <func> And <func> Relate <arg> {r2} <arg> forward'
                                                else:
                                                    tq = ".".join(q.split('.')[:-1]) + ".from"
                                                    code = f'Find <arg> {entity_name[v]} <func> Relate <arg> {merge_relation(get_name(r), get_name(r2))} <arg> forward <func> QFilterNum <arg> {merge_relation(get_name(r), get_name(q))} <arg> {entity_name[qv]}'
                                                    code_raw = f'Find <arg> {v} <func> Relate <arg> {r} <arg> forward <func> FindAll <func> FilterNum <arg> {q} <arg> {entity_name[qv]} <arg> = <func> And <func> ConstrainYear <arg> {tq} <arg> NOW <func> Relate <arg> {r2} <arg> forward'
                                                initial_programs.append(Program(source={v, qv},
                                                                code=code,
                                                                code_raw=code_raw,
                                                                function='Relate'))
                                # time constrain
                                if len(possible_years) > 0 and d == 'forward':
                                    possible_tqs = [q for q in self.get_tc_constraints_for_program(program) if q != r2]
                                    for tq in possible_tqs:
                                        for qt in possible_years:
                                            code = f'Find <arg> {entity_name[v]} <func> Relate <arg> {merge_relation(get_name(r), get_name(r2))} <arg> forward <func> QFilterYear <arg> {merge_relation(get_name(r), get_name(tq))} <arg> {entity_name[qt]} <arg> ='
                                            code_raw = f'Find <arg> {v} <func> Relate <arg> {r} <arg> forward <func> ConstrainYear <arg> {tq} <arg> {qt} <func> Relate <arg> {r2} <arg> forward'
                                            initial_programs.append(Program(source={v, qt},
                                                            code=code,
                                                            code_raw=code_raw,
                                                            function='Relate'))
                                pass
                                    
                    else:
                        initial_programs.append(program)
                        
                if self._dataset == "webq" and self.is_query_attr(question):
                    possible_attributes = self.get_attributes_for_variables({v})
                    for at in possible_attributes:
                        code = f'Find <arg> {entity_name[v]} <func> QueryAttr <arg> {get_name(at)}'
                        code_raw = f'Find <arg> {v} <func> QueryAttr <arg> {at}'
                        initial_programs.append(Program(source=v,
                                                        code=code,
                                                        code_raw=code_raw,
                                                        function='QueryAttr',
                                                        ))
                
            else:
                if self._dataset == 'webq':
                    continue
                filter_func = data_type2filter_func[v.split('#')[-1]]
                possible_attributes_v = set(self.get_attributes_for_value(v, use_ontology=False))
                for at in answer_types:
                    possible_attributes_at = self.get_attributes_for_class(at)
                    for r in possible_attributes_at:
                        if r not in possible_attributes_v: continue
                        code = f'FindAll <func> {filter_func} <arg> {get_name(r)} <arg> {entity_name[v]} <arg> = <func> FilterConcept <arg> {get_name(at)}'
                        code_raw = f'FindAll <func> {filter_func} <arg> {r} <arg> {v} <arg> = <func> FilterConcept <arg> {at}'
                        initial_programs.append(Program(source=v,
                                                        code=code,
                                                        code_raw=code_raw,
                                                        function="FilterConcept",
                                                        ))

                if self._dataset != "webq":
                    possible_attributes_v = set(self.get_attributes_for_value(v, use_ontology=False))
                    for at in answer_types:
                        possible_attributes_at = self.get_attributes_for_class(at)
                        for r in possible_attributes_at:
                            if r not in possible_attributes_v: continue
                            for comp in [">=", "<=", ">", "<"]:
                                code = f'FindAll <func> {filter_func} <arg> {get_name(r)} <arg> {entity_name[v]} <arg> {comp} <func> FilterConcept <arg> {get_name(at)}'
                                code_raw = f'FindAll <func> {filter_func} <arg> {r} <arg> {v} <arg> {comp} <func> FilterConcept <arg> {at}'
                                initial_programs.append(Program(source=v,
                                                                code=code,
                                                                code_raw=code_raw,
                                                                function="FilterConcept",
                                                                ))

        # The following is for (ARGXXX Class_Name Relation/Attribute)
        if self._dataset == 'webq':
            answer_types = []
        for at in answer_types:
            if len(entity_name) > 0:
                break
            if at in self.cvt_types:
                continue

            possible_attributes = self.get_attributes_for_class(at)
            for a in possible_attributes:
                for comp in ['smallest', 'largest']:
                    code = f'FindAll <func> FilterConcept <arg> {get_name(at)} <func> SelectAmong <arg> {get_name(a)} <arg> {comp}'
                    code_raw = f'FindAll <func> FilterConcept <arg> {at} <func> SelectAmong <arg> {a} <arg> {comp}'
                    initial_programs.append(Program(source=at,
                                                    code=code,
                                                    code_raw=code_raw,
                                                    function='SelectAmong',
                                                    ))

        return initial_programs


if __name__ == '__main__':
    computer = Computer(dataset="grail")
    program = Program(
        code_raw='FindAll <func> FilterNum <arg> measurement_unit.time_unit.time_in_seconds <arg> 1000.0^^http://www.w3.org/2001/XMLSchema#float <arg> < <func> FilterConcept <arg> measurement_unit.time_unit <func> Find <arg> m.0c13h <func> Relate <arg> measurement_unit.thermal_conductivity_unit.measurement_system <arg> backward <func> FilterConcept <arg> freebase.unit_profile'
    )
    res = computer.get_relations_for_program(program, reverse=True)
    print(res)
