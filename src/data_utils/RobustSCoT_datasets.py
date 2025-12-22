import json
import random
import os
from typing import Dict, List, Any
from dataclasses import dataclass

import torch
import transformers
from torch.utils.data import Dataset
import numpy as np

DATA_DIR = "data/processed"

def load_dataset(dataset_name: str):
    if dataset_name == "STAIR-SFT_diverse":
        data_path = os.path.join(DATA_DIR, f'STAIR-SFT_diverse.json')
        with open(data_path, 'r') as f:
            dataset = json.load(f)
        new_benign_dataset = []
        new_harmful_dataset = []
        for data in dataset:
            if data['source'] == 'PKU-SafeRLHF':
                new_harmful_dataset.append({
                    "question": data['instruction'],
                    "answer": data['cot'] + "\n" + data['answer'],
                    "mutations": data['mutations']
                })
            else:
                new_benign_dataset.append({
                    "question": data['instruction'],
                    "answer": data['cot'] + "\n" + data['answer'],
                })
        return new_benign_dataset, new_harmful_dataset
    else:
        raise ValueError(f"Dataset {dataset_name} not found")


'''Dataset for Safety Reasoning'''
class SafetyReasoningDataset(Dataset):
    
    def __init__(self, 
        dataset: List[Dict],
        tokenizer: transformers.PreTrainedTokenizer, 
        max_length: int = 2048,
    ):
        super(SafetyReasoningDataset, self).__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.dataset = dataset

    def _format_data(self, question, answer):
        # Format messages for refusal dataset and retain dataset
        messages = [
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer}
        ]
        # apply chat template
        formatted_text = self.tokenizer.apply_chat_template(messages, tokenize=False)

        # Tokenize with explicit padding and truncation
        encodings = self.tokenizer(formatted_text, 
                                   return_tensors='pt', 
                                   max_length=self.max_length, 
                                   padding='max_length',
                                   truncation=True)
        
        # Create the model inputs
        model_inputs = {
            'input_ids': encodings['input_ids'][0],  # Remove batch dimension
            'attention_mask': encodings['attention_mask'][0]
        }

        # create labels
        labels = encodings['input_ids'][0].clone()
        formatted_text_wo_assistant = self.tokenizer.apply_chat_template(messages[:-1], tokenize=False)
        encodings_wo_assistant = self.tokenizer(formatted_text_wo_assistant, 
                                                return_tensors='pt', 
                                                padding=False)
        labels[:encodings_wo_assistant.input_ids.shape[1]] = -100   
        model_inputs['labels'] = labels
        return model_inputs.copy()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        # Get items from both datasets
        item = self.dataset[i]
        question = item['question']
        answer = item['answer']
        mutation = item.get('selected_mutation', None)
        data_type = item.get('data_type', 'benign')  # 'benign' or 'harmful'
        is_adv = True if data_type == 'harmful' and item.get('selected_mutation') is not None else False

        normal_inputs = self._format_data(question, answer)

        model_inputs = dict(
            input_ids=normal_inputs['input_ids'],
            attention_mask=normal_inputs['attention_mask'],
            labels=normal_inputs['labels'],
            data_type=data_type,  # 'benign' or 'harmful'
            is_adv=torch.tensor(1 if is_adv else 0, dtype=torch.long),  # mask for adversarial
        )
        if mutation is not None and is_adv:
            adv_inputs = self._format_data(mutation, answer)
            model_inputs['adv_input_ids'] = adv_inputs['input_ids']
            model_inputs['adv_attention_mask'] = adv_inputs['attention_mask']
            model_inputs['adv_labels'] = adv_inputs['labels']
        else:
            # No adversarial mutation, use dummy values (will be masked out)
            model_inputs['adv_input_ids'] = normal_inputs['input_ids'].clone()
            model_inputs['adv_attention_mask'] = normal_inputs['attention_mask'].clone()
            model_inputs['adv_labels'] = normal_inputs['labels'].clone()
        
        return model_inputs


@dataclass
class SafetyDataCollator:
    """
    Custom data collator that handles special fields like data_type and is_adv
    """
    tokenizer: transformers.PreTrainedTokenizer
    
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        # Separate special fields from tensor fields
        data_types = [f.pop('data_type') for f in features]
        is_adv = torch.stack([f.pop('is_adv') for f in features])
        
        # Use standard collator for tensor fields
        batch = self.tokenizer.pad(
            features,
            padding=True,
            return_tensors='pt',
        )
        
        # Add back special fields
        batch['data_type'] = data_types  # Keep as list of strings
        batch['is_adv'] = is_adv  # Already a tensor
        
        return batch


if __name__ == "__main__":
    from transformers import AutoTokenizer
    dataset = json.load(open("data/processed/STAIR-SFT_diverse.json", "r"))

    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
    tokenizer.pad_token = tokenizer.eos_token
    train_dataset = SafetyReasoningDataset(
        dataset=dataset,
        tokenizer=tokenizer,
        max_length=2048,
    )
    print(train_dataset.dataset[0])

    