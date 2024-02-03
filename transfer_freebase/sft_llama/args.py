from dataclasses import dataclass, field
from typing import Optional, List, Union
from transformers import MODEL_FOR_CAUSAL_LM_MAPPING
from transformers import Seq2SeqTrainingArguments

@dataclass
class ModelArguments:
    model_name_or_path: str = field(default="facebook/opt-125m")
    compressor_name_or_path: Optional[str] = field(default=None)
    lora_r: Optional[int] = field(default=16)
    lora_alpha: Optional[int] = field(default=32)
    target_modules: Optional[str] = field(
        default='q_proj,v_proj,k_proj,o_proj,gate_proj,down_proj,up_proj',
        metadata={
            "help": "List of module names or regex expression of the module names to replace with Lora."
            "For example, ['q', 'v'] or '.*decoder.*(SelfAttention|EncDecAttention).*(q|v)$' "
        },
    )
    def __post_init__(self):
        if type(self.target_modules)==str:
            self.target_modules = self.target_modules.split(',')

@dataclass
class DataArguments:
    train_file: Optional[str] = field(default=None, metadata={"help": "Path to the training data."})
    validation_file: Optional[str] = field(default=None, metadata={"help": "Path to the training data."})
    processed_data_dir: Optional[str] = field(default=None, metadata={"help": "Path to the processed data."})
    preprocessing_num_workers: Optional[int] = field(
        default=1,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    prompt_column: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the column in the datasets containing the full texts (for summarization)."},
    )
    response_column: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the column in the datasets containing the summaries (for summarization)."},
    )
    history_column: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the column in the datasets containing the history of chat."},
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of prediction examples to this "
                "value if set."
            )
        },
    )

@dataclass
class TrainingArguments(Seq2SeqTrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    single_kb: Optional[bool] = field(default=False)
    