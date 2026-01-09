
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
    dataset_path: str = field(
        default="r2d_r1",
        metadata={"help": "Dataset name from the Hugging Face datasets library"}
    )
    max_length: int = field(
        default=2048, metadata={"help": "Maximum length of the input sequence"}
    )
    split: str = field(
        default="train", metadata={"help": "Dataset split to use"}
    )

@dataclass
class TrainingArguments(HfTrainingArguments):
    output_dir: str = field(
        default="./output",
        metadata={"help": "The output directory where the model predictions and checkpoints will be written."}
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
