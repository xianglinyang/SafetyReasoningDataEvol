import torch
from torch.utils.data import Dataset
from datasets import load_dataset
from typing import Dict, Any
import transformers

def data_reader(dataset_name="r2d_r1"):
    if dataset_name == "r2d_r1":
        return load_dataset("chuhac/R2D-R1")


class R2DDataset(Dataset):
    """
    Dataset class for R2D (Representation Rerouting for Safety) training.
    
    This dataset handles tokenization of instruction-input-output triplets,
    where only the output portion is used for loss calculation (labels).
    """
    
    def __init__(
        self,
        dataset: Any,
        tokenizer: transformers.PreTrainedTokenizer,
        max_length: int = 2048,
        split: str = "train"
    ):
        """
        Args:
            dataset: HuggingFace dataset object
            tokenizer: PreTrainedTokenizer for encoding text
            max_length: Maximum sequence length
            split: Dataset split to use (e.g., "train", "test")
        """
        super(R2DDataset, self).__init__()
        
        self.dataset = dataset[split]
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self) -> int:
        return len(self.dataset)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single tokenized example.
        
        Returns:
            dict with keys:
                - input_ids: torch.Tensor of token ids
                - labels: torch.Tensor of labels (-100 for instruction+input, token ids for output)
        """
        example = self.dataset[idx]
        # Use max_length parameter to avoid warnings
        inst_enc = self.tokenizer.encode(example["instruction"], padding=False, truncation=True, max_length=self.max_length)
        inp_enc = self.tokenizer.encode(example["input"], padding=False, truncation=True, max_length=self.max_length)
        out_enc = self.tokenizer.encode(example["output"], padding=False, truncation=True, max_length=self.max_length)
        
        input_ids = inst_enc + inp_enc + out_enc + [self.tokenizer.eos_token_id]
        labels = [-100] * len(inst_enc + inp_enc) + out_enc + [self.tokenizer.eos_token_id]

        return dict[str, list[str | list[str] | Any | int | None]](input_ids=input_ids, labels=labels)