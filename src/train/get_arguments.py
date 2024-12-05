'''Adapt from https://raw.githubusercontent.com/GraySwanAI/circuit-breakers/refs/heads/main/src/args.py'''
import logging

from dataclasses import dataclass, field
from typing import Optional, Dict, Sequence, Union, List
import transformers
import typing
logger = logging.getLogger(__name__)

@dataclass
class DataArguments:
    dataset_name: str = field(
        default="circuitbreaker",
        metadata={"help": "Name of the dataset to use"}
    )
    max_seq_length: Optional[int] = field(
        default=1024,
        metadata={
            "help": "The maximum total input sequence length after tokenization."
        }
    )

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
    model_name_abbr: str = field(
        default="llama2",
        metadata={"help": "Abbreviated name of the model (e.g., llama2, llama3)"}
    )

@dataclass
class LorraArguments:
    target_layers: str = field(metadata={"help": "Layers for Representation. Layers are seperate by `,` eg: `10,12,14,16,18,20` "})
    transform_layers: str = field(metadata={"help": "Layers for Representation. Layers are seperate by `,` eg: `10,12,14,16,18,20` "})
    lorra_alpha: float = field(default=5, metadata={"help": ""}) # LoRRA Hyperparameters
    trainsets: typing.List[str] = field(default=None, metadata={"help": "A list of trainsets for finetuning the corresponding Concepts/Functions, separated by # for commandline inputs (eg: ['AlpacaSupervisedDataset'])"})
    valsets: typing.List[str] = field(default=None, metadata={"help": "A list of valsets for finetuning the corresponding Concepts/Functions, separated by # for commandline inputs (eg: ['AlpacaSupervisedDataset'])"})
    adv_string: str = field(default="", metadata={"help": "adversarial string for harmful prompt. (eg: Start with 'Sure here's')"})
    full_layers: bool = field(
        default=False, metadata={"help": "Whether to drop not used layer during training"}
    )

@dataclass
class LoraArguments:
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target_modules: Union[List[str], str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )
    lora_weight_path: str = ""
    lora_bias: str = "none"
    q_lora: bool = False



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
class TrainingArguments(transformers.TrainingArguments):
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