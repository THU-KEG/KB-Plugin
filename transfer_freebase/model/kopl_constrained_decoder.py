from transformers import GenerationMixin
import copy
import inspect
import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union
import math
import torch
import torch.distributed as dist
from torch import nn
import json

from transformers.modeling_outputs import CausalLMOutputWithPast, Seq2SeqLMOutput
from transformers.models.auto import (
    MODEL_FOR_CAUSAL_IMAGE_MODELING_MAPPING,
    MODEL_FOR_CAUSAL_LM_MAPPING,
    MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING,
    MODEL_FOR_SPEECH_SEQ_2_SEQ_MAPPING,
    MODEL_FOR_VISION_2_SEQ_MAPPING,
)
from transformers.utils import ExplicitEnum, ModelOutput, is_accelerate_available, logging
from transformers.generation.beam_constraints import DisjunctiveConstraint, PhrasalConstraint
from transformers.generation.beam_search import BeamScorer, BeamSearchScorer, ConstrainedBeamSearchScorer
from transformers.generation.configuration_utils import GenerationConfig
from transformers.generation.logits_process import (
    LogitsProcessor,
    EncoderNoRepeatNGramLogitsProcessor,
    EncoderRepetitionPenaltyLogitsProcessor,
    EpsilonLogitsWarper,
    EtaLogitsWarper,
    ExponentialDecayLengthPenalty,
    ForcedBOSTokenLogitsProcessor,
    ForcedEOSTokenLogitsProcessor,
    ForceTokensLogitsProcessor,
    HammingDiversityLogitsProcessor,
    InfNanRemoveLogitsProcessor,
    LogitNormalization,
    LogitsProcessorList,
    MinLengthLogitsProcessor,
    MinNewTokensLengthLogitsProcessor,
    NoBadWordsLogitsProcessor,
    NoRepeatNGramLogitsProcessor,
    PrefixConstrainedLogitsProcessor,
    RepetitionPenaltyLogitsProcessor,
    SequenceBiasLogitsProcessor,
    SuppressTokensAtBeginLogitsProcessor,
    SuppressTokensLogitsProcessor,
    TemperatureLogitsWarper,
    TopKLogitsWarper,
    TopPLogitsWarper,
    TypicalLogitsWarper,
)
from transformers.generation.stopping_criteria import (
    MaxLengthCriteria,
    MaxTimeCriteria,
    StoppingCriteria,
    StoppingCriteriaList,
    validate_stopping_criteria,
)
from transformers.generation.utils import (
    BeamSearchOutput,
    BeamSearchEncoderDecoderOutput,
    BeamSearchDecoderOnlyOutput
)
from utils.kb_environment import Program, Computer
from collections import defaultdict
from copy import copy, deepcopy
from model.llama import LlamaForCausalLM
from utils.logic_form_util import parse_seq_program, get_program_seq

class ProgramState:
    def __init__(self, tokenizer, current_program, initial_programs, admissible_programs, entity_name):
        self.tokenizer = tokenizer
        self.initial_programs = defaultdict(list)
        if isinstance(initial_programs, list):
            for program in initial_programs:
                if isinstance(program.source, str):
                    self.initial_programs[program.source].append(program)
                else:
                    self.initial_programs[tuple(sorted(program.source))].append(program)
        else:
            self.initial_programs = initial_programs
        self.entity_name = entity_name
        
        self.end_token = '<end>'
        self.splitter_bos = self.tokenize('<func>')[0]
        self.eos = self.tokenizer.eos_token
        self.trie = {}
        self.build_trie(admissible_programs)
        self.cur_node = self.trie
        if current_program is not None:
            self.move_on_trie(self.tokenize(current_program.code))
    
    def tokenize(self, sentence):
        return self.tokenizer.tokenize(sentence)
    
    def add_program_to_trie(self, program):
        cur_node = self.trie
        tokens = self.tokenize(program.code)
        for depth, token in enumerate(tokens):
            if token not in cur_node:
                cur_node[token] = {}
            cur_node = cur_node[token]
            if depth == len(tokens) - 1:
                if self.end_token not in cur_node:
                    cur_node[self.end_token] = []
                cur_node[self.end_token].append(program)
    
    def build_trie(self, programs):
        for program in programs:
            self.add_program_to_trie(program)
    
    def get_next_valid_tokens(self):
        finished_programs = defaultdict(list)
        valid_tokens = list(self.cur_node.keys())
        if self.end_token in self.cur_node:
            valid_tokens.remove(self.end_token)
            for program in self.cur_node[self.end_token]:
                if program.function in ["What", "Count", "QueryAttr", "QueryRelationQualifier"]:
                    valid_tokens += [self.eos]
                    finished_programs[self.eos].append(program)
                else:
                    valid_tokens += [self.splitter_bos]
                    finished_programs[self.splitter_bos].append(program)
        return valid_tokens, finished_programs
    
    def move_on_trie(self, tokens):
        for token in tokens:
            self.cur_node = self.cur_node[token]
       
class KoPLConstrainedDecoder(LlamaForCausalLM):
    
    def __init__(self, config, tokenizer, dataset='grail'):
        super().__init__(config)
        self.tokenizer = tokenizer
        self.computer = Computer(dataset)
        self._dataset = dataset

    def initial_program_states(self, entity_names, answer_types, questions, batch_size, num_beams):
        assert len(entity_names) == batch_size and len(answer_types) == batch_size and len(questions) == batch_size
        program_states = []
        for entity_name, answer_type, question in zip(entity_names, answer_types, questions):
            initial_programs = self.computer.get_initial_programs(entity_name, answer_type, question)
            # for program in initial_programs:
            #     print(program.code)
            #     print(program.code_raw)
            #     print()
            # exit()
            program_state = ProgramState(
                tokenizer=self.tokenizer,
                current_program=None,
                initial_programs=initial_programs,
                admissible_programs=initial_programs,
                entity_name=entity_name
            )
            program_states.extend([program_state] + [None] * (num_beams - 1))
        return program_states
    
    def get_next_valid_tokens(self, input_ids, scores, program_states, batch_size, num_beams):
        mask = torch.full_like(scores, -math.inf)
        all_valid_tokens = []
        all_finished_programs = []
        for i, sent in enumerate(input_ids):
            program_state = program_states[i]
            if program_state is None:
                all_valid_tokens.append({})
                all_finished_programs.append({})
            else:
                valid_tokens, finished_programs = program_state.get_next_valid_tokens()
                all_valid_tokens.append(set(valid_tokens))
                mask[i, self.tokenizer.convert_tokens_to_ids(valid_tokens)] = 0
                all_finished_programs.append(finished_programs)
        return scores + mask, all_valid_tokens, all_finished_programs
    
    def check_finish(self, code):
        program = parse_seq_program(code)
        return program[-1]['function'] in ['What', 'Count', 'QueryAttr', 'QueryRelationQualifier']
    
    def post_process(self, program):
        kopl = []
        for x in program.kopl:
            func, inputs = x["function"], x["inputs"]
            if func == "FilterConcept" and inputs[0] not in self.computer._classes_vocab:
                continue
            # if func == "Relate" and inputs == ['book.written_work.author', 'backward']:
            #     x["inputs"] = ['book.author.book_editions_published', 'forward']
            # if func == "FilterConcept" and kopl[-1]["inputs"] == ['book.author.book_editions_published', 'forward']:
            #     continue
            kopl.append(x)
        program.kopl = kopl
        program.code_raw = get_program_seq(kopl)
        return program
    
    def beam_search(
        self,
        input_ids: torch.LongTensor,
        beam_scorer: BeamScorer,
        logits_processor: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        max_length: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[Union[int, List[int]]] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_scores: Optional[bool] = None,
        return_dict_in_generate: Optional[bool] = None,
        synced_gpus: bool = False,
        **model_kwargs,
    ) -> Union[BeamSearchOutput, torch.LongTensor]:
        
        # init values
        logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
        stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()
        if max_length is not None:
            warnings.warn(
                "`max_length` is deprecated in this function, use"
                " `stopping_criteria=StoppingCriteriaList(MaxLengthCriteria(max_length=max_length))` instead.",
                UserWarning,
            )
            stopping_criteria = validate_stopping_criteria(stopping_criteria, max_length)
        if len(stopping_criteria) == 0:
            warnings.warn("You don't have defined any stopping_criteria, this will likely loop forever", UserWarning)
        pad_token_id = pad_token_id if pad_token_id is not None else self.generation_config.pad_token_id
        eos_token_id = eos_token_id if eos_token_id is not None else self.generation_config.eos_token_id
        if isinstance(eos_token_id, int):
            eos_token_id = [eos_token_id]
        output_scores = output_scores if output_scores is not None else self.generation_config.output_scores
        output_attentions = (
            output_attentions if output_attentions is not None else self.generation_config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.generation_config.output_hidden_states
        )
        return_dict_in_generate = (
            return_dict_in_generate
            if return_dict_in_generate is not None
            else self.generation_config.return_dict_in_generate
        )

        batch_size = len(beam_scorer._beam_hyps)
        num_beams = beam_scorer.num_beams

        batch_beam_size, cur_len = input_ids.shape

        if num_beams * batch_size != batch_beam_size:
            raise ValueError(
                f"Batch dimension of `input_ids` should be {num_beams * batch_size}, but is {batch_beam_size}."
            )

        # init attention / hidden states / scores tuples
        scores = () if (return_dict_in_generate and output_scores) else None
        beam_indices = (
            tuple(() for _ in range(batch_beam_size)) if (return_dict_in_generate and output_scores) else None
        )
        decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
        cross_attentions = () if (return_dict_in_generate and output_attentions) else None
        decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

        # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
        if return_dict_in_generate and self.config.is_encoder_decoder:
            encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
            encoder_hidden_states = (
                model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None
            )

        # initialise score of first beam with 0 and the rest with -1e9. This makes sure that only tokens
        # of the first beam are considered to avoid sampling the exact same tokens across all beams.
        beam_scores = torch.zeros((batch_size, num_beams), dtype=torch.float, device=input_ids.device)
        beam_scores[:, 1:] = -1e9
        beam_scores = beam_scores.view((batch_size * num_beams,))
        
        entity_name, answer_types, question = model_kwargs["entity_name"], model_kwargs["answer_types"], model_kwargs["question"]
        program_states = self.initial_program_states(entity_name, answer_types, question, batch_size, num_beams)

        this_peer_finished = False  # used by synced_gpus only
        
        code2programs = [defaultdict(list) for i in range(batch_size)]
        
        while True:
            if synced_gpus:
                # Under synced_gpus the `forward` call must continue until all gpus complete their sequence.
                # The following logic allows an early break if all peers finished generating their sequence
                this_peer_finished_flag = torch.tensor(0.0 if this_peer_finished else 1.0).to(input_ids.device)
                # send 0.0 if we finished, 1.0 otherwise
                dist.all_reduce(this_peer_finished_flag, op=dist.ReduceOp.SUM)
                # did all peers finish? the reduced sum will be 0.0 then
                if this_peer_finished_flag.item() == 0.0:
                    break

            model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)

            outputs = self(
                **model_inputs,
                return_dict=True,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                entity_name=entity_name,
                answer_types=answer_types,
                question=question,
            )

            if synced_gpus and this_peer_finished:
                cur_len = cur_len + 1
                continue  # don't waste resources running the code we don't need

            next_token_logits = outputs.logits[:, -1, :]
            next_token_scores = nn.functional.log_softmax(
                next_token_logits, dim=-1
            )  # (batch_size * num_beams, vocab_size)

            next_token_scores_processed = logits_processor(input_ids, next_token_scores)
            
            next_token_scores_processed, all_valid_tokens, all_finished_programs = self.get_next_valid_tokens(input_ids, next_token_scores_processed, program_states, batch_size, num_beams)
            
            next_token_scores = next_token_scores_processed + beam_scores[:, None].expand_as(
                next_token_scores_processed
            )

            # Store scores, attentions and hidden_states when required
            if return_dict_in_generate:
                if output_scores:
                    scores += (next_token_scores_processed,)
                if output_attentions:
                    decoder_attentions += (
                        (outputs.decoder_attentions,) if self.config.is_encoder_decoder else (outputs.attentions,)
                    )
                    if self.config.is_encoder_decoder:
                        cross_attentions += (outputs.cross_attentions,)

                if output_hidden_states:
                    decoder_hidden_states += (
                        (outputs.decoder_hidden_states,)
                        if self.config.is_encoder_decoder
                        else (outputs.hidden_states,)
                    )

            # reshape for beam search
            vocab_size = next_token_scores.shape[-1]
            next_token_scores = next_token_scores.view(batch_size, num_beams * vocab_size)

            # Sample 1 + len(eos_token_id) next tokens for each beam so we have at least 1 non eos token per beam.
            n_eos_tokens = len(eos_token_id) if eos_token_id else 0
            next_token_scores, next_tokens = torch.topk(
                next_token_scores, max(2, 1 + n_eos_tokens) * num_beams, dim=1, largest=True, sorted=True
            )

            next_indices = torch.div(next_tokens, vocab_size, rounding_mode="floor")
            next_tokens = next_tokens % vocab_size
            
            for batch_idx, (next_indices_i, next_tokens_i) in enumerate(zip(next_indices, next_tokens)):
                for next_indice, next_token in zip(next_indices_i, next_tokens_i):
                    if next_token == self.tokenizer.eos_token_id:
                        idx = batch_idx * num_beams + next_indice
                        finished_programs = all_finished_programs[idx].get(self.tokenizer.eos_token, [])
                        
                        for program in finished_programs:
                            code2programs[batch_idx][program.code].append(program)
                        
            beam_outputs = beam_scorer.process(
                input_ids,
                next_token_scores,
                next_tokens,
                next_indices,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
                beam_indices=beam_indices,
            )

            beam_scores = beam_outputs["next_beam_scores"]
            beam_next_tokens = beam_outputs["next_beam_tokens"]
            beam_idx = beam_outputs["next_beam_indices"]
            # print()
            # print(len(input_ids[0]))
            # print(beam_scores)
            # print(beam_next_tokens)
            # print(self.tokenizer.convert_ids_to_tokens(beam_next_tokens))
            # print(beam_idx)
            new_program_states = []
            for idx, next_token in zip(beam_idx, self.tokenizer.convert_ids_to_tokens(beam_next_tokens)):
                program_state = program_states[idx]
                valid_tokens = all_valid_tokens[idx]
                # print(self.tokenizer.decode(input_ids[idx], skip_special_tokens=True).split('\n')[1].strip())
                # print(next_token, valid_tokens)
                if next_token not in valid_tokens:
                    program_state = None
                else:
                    finished_programs = all_finished_programs[idx].get(next_token, [])
                    if len(finished_programs) == 0:
                        program_state = copy(program_state)
                        program_state.move_on_trie([next_token])
                    else:
                        admissible_programs = []
                        # print(len(finished_programs))
                        for program in (finished_programs if self._dataset not in ["gq1"] else self.computer.choose_final_program(finished_programs, k=10)):#[:5]:
                            # print()
                            # print(program.code)
                            # print(program.code_raw)
                            # print(program.kopl)
                            admissible_programs.extend(
                                self.computer.get_admissible_programs(
                                    program=program, 
                                    initial_programs=program_state.initial_programs,
                                    entity_name=program_state.entity_name,
                                    question=question[idx//num_beams]
                                )
                            )
                        # print(len(admissible_programs))
                        # for x in admissible_programs:
                        #     print(x.code_raw)
                        #     print(x.code)
                        #     print()
                        if len(admissible_programs) == 0:
                            program_state = None
                        else:
                            program_state = ProgramState(
                                tokenizer=self.tokenizer,
                                current_program=program,
                                initial_programs=program_state.initial_programs,
                                admissible_programs=admissible_programs,
                                entity_name=program_state.entity_name
                            )
                            program_state.move_on_trie([next_token])
                    
                new_program_states.append(program_state)
                
            program_states = new_program_states

            input_ids = torch.cat([input_ids[beam_idx, :], beam_next_tokens.unsqueeze(-1)], dim=-1)

            model_kwargs = self._update_model_kwargs_for_generation(
                outputs, model_kwargs, is_encoder_decoder=self.config.is_encoder_decoder
            )
            if model_kwargs["past_key_values"] is not None:
                model_kwargs["past_key_values"] = self._reorder_cache(model_kwargs["past_key_values"], beam_idx)

            if return_dict_in_generate and output_scores:
                beam_indices = tuple((beam_indices[beam_idx[i]] + (beam_idx[i],) for i in range(len(beam_indices))))

            # increase cur_len
            cur_len = cur_len + 1

            if beam_scorer.is_done or stopping_criteria(input_ids, scores):
                if not synced_gpus:
                    break
                else:
                    this_peer_finished = True

        sequence_outputs = beam_scorer.finalize(
            input_ids,
            beam_scores,
            next_tokens,
            next_indices,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            max_length=stopping_criteria.max_length,
            beam_indices=beam_indices,
        )
        
        output_codes = [output.split('\n')[1].strip() for output in self.tokenizer.batch_decode(sequence_outputs["sequences"], skip_special_tokens=True)]
        num_return_sequences = beam_scorer.num_beam_hyps_to_keep
        assert len(output_codes) == len(code2programs) * num_return_sequences
        
        output_programs = []
        for i in range(batch_size):
            code2program_map = code2programs[i]
            finish = False
            for code in output_codes[i*num_return_sequences:(i+1)*num_return_sequences]:
                if self.check_finish(code) and code in code2program_map:
                    program = self.computer.choose_final_program(code2program_map[code])
                    if self._dataset == "webq":
                        program = self.post_process(program)
                    program.execution = self.computer.execute_program(program)
                    if len(program.execution) > 0:
                        output_programs.append(program)
                        finish = True
                        break
                    else:
                        print("empty result: ", program.code_raw)
            # for code in output_codes[i*num_return_sequences:(i+1)*num_return_sequences]:
            #     # print(code)
            #     if self.check_finish(code) and code in code2program_map:
            #         program = self.computer.choose_final_program(code2program_map[code])
            #         print(program.code)
            #         print(program.code_raw)
            if not finish:
                output_programs.append("Exceed Max Length: %s" % output_codes[i*num_return_sequences])
        assert len(output_programs) == batch_size
        return output_programs
    
    def forward(self, entity_name, answer_types, question, **model_inputs):
        return super().forward(**model_inputs)