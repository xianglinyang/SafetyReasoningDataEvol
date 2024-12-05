import os
import json
from typing import Dict

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
                model_name: str     # 'llama2','llama3','mistral', 'qwen', 
                ):
        super(SafetyReasoningDataset, self).__init__()
        self.tokenizer = tokenizer
        self.model_name = model_name.lower()
        self.max_length = 1024

        # ================ Model and Template Config  ================
        assert self.model_name in MODEL_CONFIGS.keys(), f"Model {self.model_name} not supported"

        configs = MODEL_CONFIGS[self.model_name]
        self.user_tag = configs['user_tag']
        self.assistant_tag = configs['assistant_tag']
        self.system = configs['system']
        self.sep_token = configs['sep_token']

        # ======================= Load Dataset ======================= #
        PROCESSED_DATA_DIR = "data/processed"
        processed_data_path = os.path.join(PROCESSED_DATA_DIR, f'{dataset_name}_{split}.json')

        with open(processed_data_path, 'r') as file:
            dataset = json.load(file)
        
        formatted_dataset = []
        for data in dataset:
            question = data['evolved_question']
            answer = data['evolved_answer']

            formatted_input = concat_one_shot(
                instruction=question, 
                response=answer, 
                system=self.system, 
                user_tag=self.user_tag, 
                assistant_tag=self.assistant_tag)

            formatted_dataset.append(formatted_input)
        
        self.dataset = formatted_dataset

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        tokenize_kwargs = dict(max_length=1024, padding="max_length", truncation=True, return_tensors="pt")
        formatted_input = self.dataset[i]

        tokenized_inputs = self.tokenizer(formatted_input.replace('<SEPARATOR>', self.sep_token), **tokenize_kwargs)
        return tokenized_inputs