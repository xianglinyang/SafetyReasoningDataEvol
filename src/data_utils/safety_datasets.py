import json
from typing import Dict, List

import torch
import transformers
from torch.utils.data import Dataset

'''Dataset for Safety Reasoning'''
class SafetyReasoningDataset(Dataset):
    
    def __init__(self, 
                dataset_name: str,      # 'circuitbreaker'
                split: str,             # train/val/test
                tokenizer: transformers.PreTrainedTokenizer, 
                max_length: int = 2048,
                system_inst=None
                ):
        super(SafetyReasoningDataset, self).__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.dataset = self._load_data(dataset_name, split)
        self.system_inst = system_inst

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        example = self.dataset[i]
        
        question = example['evolved_question']
        answer = example['evolved_answer']

        # format data
        messages = list()
        if self.system_inst is not None:
            messages.append({"role": "system", "content": self.system_inst})
        messages.append({"role": "user", "content": question})
        messages.append({"role": "assistant", "content": answer})

        # Tokenize with explicit padding and truncation
        encodings = self.tokenizer.apply_chat_template(messages,
                                                  max_length=self.max_length,
                                                  padding='max_length',
                                                  truncation=True,
                                                  return_tensors="pt",
                                                  return_attention_mask=True,
                                                  return_dict=True)
        
        # Create the model inputs
        model_inputs = {
            'input_ids': encodings['input_ids'][0],  # Remove batch dimension
            'attention_mask': encodings['attention_mask'][0]
        }
        # Create labels (same as input_ids but with -100 for non-assistant tokens)
        labels = encodings['input_ids'][0].clone()

        # assistant_tokens_mask
        encodings_wo_assistant = self.tokenizer.apply_chat_template(messages[:-1],
                                                  max_length=self.max_length,
                                                  padding='max_length',
                                                  truncation=True,
                                                  return_tensors="pt",
                                                  return_attention_mask=True,
                                                  return_dict=True)

        # mask the prompt part for avoiding loss
        labels[:encodings_wo_assistant.input_ids.shape[1]] = -100    
    
        model_inputs['labels'] = labels
        return model_inputs
    
    def _find_subsequence(self, seq: List[int], subseq: List[int]) -> int:
        """Find the starting position of a subsequence in a sequence."""
        n, m = len(seq), len(subseq)
        for i in range(n - m + 1):
            if seq[i:i+m] == subseq:
                return i
        return None
    

    def _load_data(self, dataset_name, split):
        # Implement your data loading logic here
        data_path = f"data/processed/{dataset_name}_{split}.json"
        with open(data_path, 'r') as f:
            return json.load(f)