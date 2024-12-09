import json
from typing import Dict, List

import torch
from torch.utils.data import Dataset
import transformers

from src.train.data_utils.model_configs import MODEL_CONFIGS

'''template for one-shot and multi round messages'''

def concat_one_shot(instruction, response, system, user_tag, assistant_tag):
    if system is not None:
        one_shot_template = "{system}{user_tag}{instruction}{assistant_tag}<SEPARATOR>{response}"
        return one_shot_template.format(
            system=system,
            user_tag=user_tag, 
            instruction=instruction, 
            assistant_tag=assistant_tag,
            response=response)
    else:
        one_shot_template = "{user_tag}{instruction}{assistant_tag}<SEPARATOR>{response}"
        return one_shot_template.format(
            user_tag=user_tag, assistant_tag=assistant_tag,
            instruction=instruction, response=response)

# TODO need to verify this
def concat_messages(messages, eos_token, user_tag, assistant_tag, system_tag):
    """Concatenate messages with role delimiters and proper EOS tokens."""
    message_text = ""
    for message in messages:
        role = message["role"]
        content = message["content"].strip()
        if role == "user":
            message_text += user_tag + content + "\n"
        elif role == "assistant":
            message_text += assistant_tag + content + eos_token + "\n"
        elif role == "system":
            message_text += system_tag + content + "\n"
        else:
            raise ValueError(f"Invalid role: {role}")
    return message_text


'''Dataset for Safety Reasoning'''
class SafetyReasoningDataset(Dataset):
    
    def __init__(self, 
                dataset_name: str,      # 'circuitbreaker'
                split: str,             # train/val/test
                tokenizer: transformers.PreTrainedTokenizer, 
                model_name: str,     # 'llama2','llama3','mistral', 'qwen', 
                max_length: int = 2048
                ):
        super(SafetyReasoningDataset, self).__init__()
        self.tokenizer = tokenizer
        self.model_name = model_name.lower()
        self.max_length = max_length
        self.dataset = self._load_data(dataset_name, split)

        # ================ Model and Template Config  ================
        assert self.model_name in MODEL_CONFIGS.keys(), f"Model {self.model_name} not supported"

        configs = MODEL_CONFIGS[self.model_name]
        self.user_tag = configs['user_tag']
        self.assistant_tag = configs['assistant_tag']
        self.system = configs['system']
        self.sep_token = configs['sep_token']

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        example = self.dataset[i]
        
        question = example['evolved_question']
        answer = example['evolved_answer']

        # Format using one-shot template
        text = concat_one_shot(
            instruction=question,
            response=answer,
            system=self.system,
            user_tag=self.user_tag,
            assistant_tag=self.assistant_tag
        )
        text = text.replace('<SEPARATOR>', self.sep_token)

        # Tokenize with explicit padding and truncation
        encodings = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors="pt",
            return_attention_mask=True
        )
        
        # Create the model inputs
        model_inputs = {
            'input_ids': encodings['input_ids'][0],  # Remove batch dimension
            'attention_mask': encodings['attention_mask'][0]
        }
        # Create labels (same as input_ids but with -100 for non-assistant tokens)
        labels = encodings['input_ids'][0].clone()
        
        # Find where the assistant response starts
        assistant_tokens = self.tokenizer.encode(
            self.assistant_tag, 
            add_special_tokens=False
        )
        
        # Find the position where assistant's response starts
        response_start = self._find_subsequence(
            labels.tolist(), 
            assistant_tokens
        )
        
        if response_start is not None:
            # Mask everything before the assistant's response
            labels[:response_start + len(assistant_tokens)] = -100
            
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