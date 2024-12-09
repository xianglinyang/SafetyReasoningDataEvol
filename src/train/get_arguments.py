'''Adapt from https://raw.githubusercontent.com/GraySwanAI/circuit-breakers/refs/heads/main/src/args.py'''
import logging
import os

from dataclasses import dataclass, field
from typing import Optional, List
from transformers import TrainingArguments

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
        # default=None,
        # TODO: change to None in the future
        default="meta-llama/Llama-2-7b-chat-hf",
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
class TrainingArguments(TrainingArguments):
    """
    Custom training arguments that extend HuggingFace's TrainingArguments
    """
    # Dataset arguments
    train_dataset_name: str = field(
        default=None,
        metadata={
            "help": "The dataset to use for training."
        },
    )
    eval_dataset_name: str = field(
        default="bbh",
        metadata={
            "help": "The dataset to use for evaluation."
        },
    )

    # Output and logging arguments
    output_dir: str = field(
        default="outputs/default",
        metadata={
            "help": "The output directory where model predictions and checkpoints will be written."
        },
    )
    logging_dir: str = field(
        default=None,  # Will default to output_dir/runs
        metadata={
            "help": "Directory for storing logs. If None, defaults to output_dir/runs"
        },
    )
    logging_strategy: str = field(
        default="steps",
        metadata={
            "help": "The logging strategy to adopt during training.",
            "choices": ["no", "steps", "epoch"]
        },
    )
    logging_steps: int = field(
        default=100,
        metadata={
            "help": "Log every X updates steps. Should be used with logging_strategy='steps'"
        },
    )
    logging_first_step: bool = field(
        default=False,
        metadata={
            "help": "Log the first step metrics."
        },
    )
    report_to: str = field(
        default="tensorboard",
        metadata={
            "help": "The list of integrations to report the results and logs to.",
            "choices": ["azure_ml", "comet_ml", "mlflow", "neptune", "tensorboard", "clearml", "wandb"]
        },
    )

    def __post_init__(self):
        # Set logging_dir if not provided
        if self.logging_dir is None:
            self.logging_dir = os.path.join(self.output_dir, "runs")
        
        # Handle FSDP config
        if isinstance(self.fsdp_config, str):
            self.fsdp_config = fsdp_config[self.fsdp_config]
        
        super().__post_init__()