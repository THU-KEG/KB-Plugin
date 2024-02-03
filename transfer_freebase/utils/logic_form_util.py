from .sparql_executer import *
from copy import deepcopy

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

def get_program_seq(program):
    if program is None:
        return []
    seq = []
    for x in program:
        func, inputs = x["function"], x["inputs"]
        args = ''
        for input in inputs:
            args += ' <arg> ' + input
        seq.append(func + args)
    seq = ' <func> '.join(seq)
    return seq

def kopl_to_sparql(kopl, return_entities=False):
    kopl = deepcopy(kopl)
    clauses = []
    out_vars = []
    branch_out_vars = []
    entities = set()
    for i, item in enumerate(kopl):
        assert len(out_vars) == i
        func, inputs = item["function"], item["inputs"]
        if func == "Find":
            clauses.append([])
            clauses[-1].append("VALUES %s { :%s } " % (f"?x{i}", inputs[0]))
            entities.add(inputs[0])
            if len(out_vars) > 0:
                branch_out_vars.append(out_vars[-1])
            out_vars.append(f"?x{i}")
            
        elif func == "FindAll":
            clauses.append([])
            if len(out_vars) > 0:
                branch_out_vars.append(out_vars[-1])
            out_vars.append(f"?x{i}")
            
        elif func == "FilterConcept":
            clauses[-1].append("%s :type.object.type :%s . " % (out_vars[i-1], inputs[0]))
            out_vars.append(out_vars[i-1])
            
        elif func == "Relate":
            if inputs[1] == "forward":
                clauses[-1].append("%s :%s %s . " % (out_vars[i-1], inputs[0], f"?x{i}"))
            else:
                clauses[-1].append("%s :%s %s . " % (f"?x{i}", inputs[0], out_vars[i-1]))
            out_vars.append(f"?x{i}")

        elif func == "QueryAttr":
            clauses[-1].append("%s :%s %s . " % (out_vars[i-1], inputs[0], f"?x{i}"))
            out_vars.append(f"?x{i}")
            
        elif func in ["FilterNum", "FilterYear", "FilterDate"]:
            if '#' not in inputs[1]: # webq rawstring
                inputs[1] = f'"{inputs[1]}"'
            else:
                data_type = inputs[1].split("^^")[1].split("#")[1]
                if data_type not in ['integer', 'float', 'dateTime', 'double']:
                    inputs[1] = f'"{inputs[1].split("^^")[0] + "-08:00"}"^^<{inputs[1].split("^^")[1]}>'
                else:
                    inputs[1] = f'"{inputs[1].split("^^")[0]}"^^<{inputs[1].split("^^")[1]}>'
            if inputs[2] == "=":
                clauses[-1].append("%s :%s %s . " % (out_vars[i-1], inputs[0], inputs[1]))
            else:
                clauses[-1].append("%s :%s %s . " % (out_vars[i-1], inputs[0], f"?x{i}"))
                clauses[-1].append("FILTER (%s %s %s)" % (f"?x{i}", inputs[2], inputs[1]))
            out_vars.append(out_vars[i-1])
        
        elif func == "ConstrainYear":
            tq, qt = inputs # r, year
            assert tq.endswith(".from") or tq.endswith(".from_date")
            tq_st, tq_ed = tq, (tq[:-4]+"to") if tq.endswith(".from") else (tq[:-9]+"to_date")
            if qt == "NOW":
                qt_st = '"2015-08-10"^^xsd:dateTime'
                qt_ed = '"2015-08-10"^^xsd:dateTime'
            else:
                qt_st = f'"{qt}-01-01"^^xsd:dateTime'
                qt_ed = f'"{qt}-12-31"^^xsd:dateTime'
            clauses[-1].append(f'FILTER(\nNOT EXISTS {{{out_vars[i-1]} :{tq_st} ?sk0}} || ')
            clauses[-1].append(f'EXISTS {{{out_vars[i-1]} :{tq_st} ?sk1 . ')
            clauses[-1].append(f'FILTER(xsd:datetime(?sk1) <= {qt_ed}) }}\n)')
            clauses[-1].append(f'FILTER(\nNOT EXISTS {{{out_vars[i-1]} :{tq_ed} ?sk2}} || ')
            clauses[-1].append(f'EXISTS {{{out_vars[i-1]} :{tq_ed} ?sk3 . ')
            clauses[-1].append(f'FILTER(xsd:datetime(?sk3) >= {qt_st}) }}\n)')
            out_vars.append(out_vars[i-1])
        
        elif func == "And":
            for j in range(len(clauses[-1])):
                clauses[-1][j] = clauses[-1][j].replace(out_vars[i-1], branch_out_vars[-1])
            clauses[-2].extend(clauses[-1])
            clauses = clauses[:-1]
            out_vars.append(branch_out_vars[-1])
            branch_out_vars = branch_out_vars[:-1]
     
        elif func == "SelectAmong":
            arg_clauses = deepcopy(clauses)
            arg_clauses[-1].append("%s :%s %s . " % (out_vars[i-1], inputs[0], f"?v{i}"))
            clauses[-1].append('{\nSELECT DISTINCT (%s(%s) AS %s) WHERE {\n' % ("MAX" if inputs[1] == "largest" else "MIN", f"?v{i}", f"?x{i}") + '\n'.join(arg_clauses[-1]) + '\n}\n}')
            clauses[-1].append("%s :%s %s . " % (out_vars[i-1], inputs[0], f"?x{i}"))
            out_vars.append(out_vars[i-1])
            
        elif func == "Count":
            clauses[-1] = ['{\nSELECT COUNT(DISTINCT %s) AS %s WHERE {\n' % (out_vars[i-1], f"?x{i}") + '\n'.join(clauses[-1]) + '\n}\n}']
            out_vars.append(f"?x{i}")
        
        elif func == "What":
            out_vars.append(out_vars[i-1])
            
        else:
            print(kopl, func)
            raise NotImplementedError(func)
    
    # assert len(clauses) == 1, kopl
    for entity in entities:
        clauses[-1].append('FILTER (%s != :%s)' % (out_vars[-1], entity))
    prefix = 'PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#> PREFIX : <http://rdf.freebase.com/ns/> \n'
    sparql = prefix + 'SELECT DISTINCT (%s AS ?x) WHERE {\n' % (out_vars[-1]) + '\n'.join(clauses[-1]) + '\n}'
    if return_entities:
        return sparql, entities
    return sparql

def get_result_for_kopl(kopl):
    sparql = kopl_to_sparql(kopl)
    res = execute_query(sparql)
    return res
