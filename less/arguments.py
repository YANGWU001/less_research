import logging
import torch
from typing import List, Optional, Union
from transformers import GPT2Tokenizer, GPTNeoXTokenizerFast, LlamaTokenizer
from dataclasses import asdict, dataclass, field, fields
from transformers import TrainingArguments as TA

logger = logging.getLogger(__name__)


def none_or_str(value):
    print(value)
    if value == "None":
        return None
    else:
        return value


@dataclass
class DataArguments:
    train_files: List[str] = field(default_factory=list, metadata={
                                   "help": "The input training data files (multiple files in glob format)."})
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    max_seq_length: Optional[int] = field(
        default=None,
        metadata={
            "help": ("The maximum total input sequence length after tokenization. Sequences longer than this will be truncated,")
        },
    )
    sample_data_seed: int = field(
        default=42, metadata={"help": ("The seed used for data sampling.")},
    )
    percentage: float = field(
        default=1.0, metadata={"help": ("Sampling percentage for each dataset")},
    )


def get_data_statistics(lm_datasets):
    """ Get the data statistics of the dataset. """
    def get_length(examples):
        lengths = [len(ids) for ids in examples["input_ids"]]

        completion_lens = []
        for labels in examples["labels"]:
            com_len = (torch.tensor(labels) > -1).sum()
            completion_lens.append(com_len)
        return {"length": lengths, "c_length": completion_lens}

    if not isinstance(lm_datasets, dict):
        lm_datasets = {"train": lm_datasets}

    for key in lm_datasets:
        dataset = lm_datasets[key]
        data_size = len(dataset)
        dataset = dataset.map(get_length, batched=True)
        lengths = dataset["length"]
        length = sum(lengths) / len(lengths)
        c_lengths = dataset["c_length"]
        c_length = sum(c_lengths) / len(c_lengths)
        print(
            f"[{key} set] examples: {data_size}; # avg tokens: {length}")
        print(
            f"[{key} set] examples: {data_size}; # avg completion tokens: {c_length}")



logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The model checkpoint for weights initialization. Don't set if you want to train a model from scratch."
            )
        },
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=False,
        metadata={
            "help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={
            "help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": (
                "Will use the token generated when running `huggingface-cli login` (necessary to use this script "
                "with private models)."
            )
        },
    )
    torch_dtype: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Override the default `torch.dtype` and load the model under this dtype. If `auto` is passed, the "
                "dtype will be automatically derived from the model's weights."
            ),
            "choices": ["auto", "bfloat16", "float16", "float32"],
        },
    )

    ### added ####
    lora: Optional[bool] = field(default=False, metadata={
                                 "help": "whether to use lora"})
    lora_r: Optional[int] = field(default=8, metadata={"help": ("r for lora")})
    lora_alpha: Optional[float]=field(default=32, metadata={"help": ("alpha for lora")})
    lora_dropout: Optional[float]=field(default=0.1, metadata={"help": ("dropout for lora")})
    lora_target_modules: List[str]=field(
        default_factory=list, metadata={"help": ("target modules for lora")})


def add_padding_to_tokenizer(tokenizer):
    """ add the padding tokens in the tokenizer """
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "<pad>"})

from transformers.utils import logging
logger = logging.get_logger(__name__)
log_levels = logging.get_log_levels_dict().copy()
trainer_log_levels = dict(**log_levels, passive=-1)

fsdp_config = {
    "mpt7b_finetune": {
        "fsdp_transformer_layer_cls_to_wrap": ["MPTBlock"],
        "fsdp_backward_prefetch": "backward_pre",
        "limit_all_gathers": "true",
    },
    "opt125m_finetune": {
        "fsdp_transformer_layer_cls_to_wrap": ["OPTDecoderLayer"],
        "fsdp_backward_prefetch": "backward_pre",
        "limit_all_gathers": "true",
    },
    "mpt7b_lora": {
        "fsdp_transformer_layer_cls_to_wrap": ["MPTBlock"],
        "fsdp_backward_prefetch": "backward_pre",
        "limit_all_gathers": "true",
        "use_orig_params": "true",
    },
    "llama_finetune": {
        "fsdp_transformer_layer_cls_to_wrap": ["LlamaDecoderLayer"],
        "fsdp_backward_prefetch": "backward_pre",
        "limit_all_gathers": "true",
        "use_orig_params": "true",
    },
    "llama2_7b_finetune": {
        "fsdp_transformer_layer_cls_to_wrap": ["LlamaDecoderLayer"],
        "fsdp_backward_prefetch": "backward_pre",
        "limit_all_gathers": "true",
        "use_orig_params": "true",
    },
    "llama2_13b_finetune": {
        "fsdp_transformer_layer_cls_to_wrap": ["LlamaDecoderLayer"],
        "fsdp_backward_prefetch": "backward_pre",
        "limit_all_gathers": "true",
        "use_orig_params": "true",
    },
    "mistral_7b_finetune": {
        "fsdp_transformer_layer_cls_to_wrap": ["MistralDecoderLayer"],
        "fsdp_backward_prefetch": "backward_pre",
        "limit_all_gathers": "true",
        "use_orig_params": "true",
    },
}


@dataclass
class TrainingArguments(TA):
    analysis_mode: float = field(
        default=False,
        metadata={
            "help": (
                "Whether to run in analysis mode. "
            )
        },
    )
    analysis_dataset: str = field(
        default="bbh",
        metadata={
            "help": (
                "The dataset to use for analysis mode. "
            )
        },
    )
    train_dataset_names: str = field(
        default=None,
        metadata={
            "help": (
                "The dataset to use for training. "
            )
        },
    )


    def __post_init__(self):
        if isinstance(self.fsdp_config, str):
            self.fsdp_config = fsdp_config[self.fsdp_config]
        if self.train_dataset_names is not None:
            self.train_dataset_names = self.train_dataset_names.split(" ")
        super().__post_init__()