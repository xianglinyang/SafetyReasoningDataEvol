
from dataclasses import dataclass, field
from typing import Optional, List
from transformers import TrainingArguments as HfTrainingArguments

@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        metadata={"help": "Path to pre-trained model or model identifier from huggingface.co/models"}
    )
    use_lora: bool = field(
        default=False, metadata={"help": "Use LoRA or not"}
    )

@dataclass
class DataArguments:
    dataset_name: str = field(
        metadata={"help": "Dataset name from the Hugging Face datasets library"}
    )
    max_length: int = field(
        default=2048, metadata={"help": "Maximum length of the input sequence"}
    )

@dataclass
class TrainingArguments(HfTrainingArguments):
    output_dir: str = field(
        metadata={"help": "The output directory where the model predictions and checkpoints will be written."}
    )
    lr: float = field(
        default=2e-4, metadata={"help": "Learning rate"}
    )
    weight_decay: float = field(
        default=0.00, metadata={"help": "Weight decay"}
    )
    warmup_ratio: float = field(
        default=0.03, metadata={"help": "Warmup ratio"}
    )
@dataclass
class LoraArguments:
    r: int = field(
        default=8, metadata={"help": "Rank parameter for LoRA"}
    )
    lora_alpha: float = field(
        default=8, metadata={"help": "Hyper Alpha for LoRA"}
    )
    modules_to_save: Optional[List[str]] = field(
        default=None, 
    )
    target_modules: Optional[List[str]] = field(
        default=None, 
    )
    lora_dropout: float = field(
        default=0.05, metadata={"help": "The dropout probability for Lora layers."}
    )
    bias: str = field(
        default="none", metadata={"help": "bias"}
    )

@dataclass
class OrthSAMArguments:
    rho: float = field(
        default=0.05, metadata={"help": "SAM radius"}
    )
    adaptive: bool = field(
        default=False, metadata={"help": "Adaptive SAM"}
    )
    lam_u: float = field(
        default=2.0, metadata={"help": "Lambda for retain loss weighting"}
    )
    eps: float = field(
        default=1e-12, metadata={"help": "Epsilon for SAM"}
    )
    orth_sam_max_grad_norm: Optional[float] = field(
        default=None, metadata={"help": "Max gradient norm for SAM"}
    )
    proj_scale: float = field(
        default=1.0, metadata={"help": "Projection scale for SAM"}
    )