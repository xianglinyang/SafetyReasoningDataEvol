import json
import random
import os
from typing import Dict, List

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
        new_dataset = []
        for data in dataset:
            new_dataset.append({
                "question": data['instruction'],
                "answer": data['cot'] + "\n" + data['answer'],
                "mutations": data['mutations']
            })
        return new_dataset
    else:
        raise ValueError(f"Dataset {dataset_name} not found")


'''Dataset for Safety Reasoning'''
class SafetyReasoningDataset(Dataset):
    
    def __init__(self, 
        dataset: List[Dict],
        model_name: str,         # 'llama3'
        tokenizer: transformers.PreTrainedTokenizer, 
        max_length: int = 2048,
    ):
        super(SafetyReasoningDataset, self).__init__()
        self.model_name = model_name
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
        mutation = item['mutation'] if 'mutation' in item else None
        normal_inputs = self._format_data(question, answer)

        model_inputs = dict(
            input_ids=normal_inputs['input_ids'],
            attention_mask=normal_inputs['attention_mask'],
            labels=normal_inputs['labels'],
        )
        if mutation is not None:
            adv_inputs = self._format_data(mutation, answer)
            model_inputs['adv_input_ids'] = adv_inputs['input_ids']
            model_inputs['adv_attention_mask'] = adv_inputs['attention_mask']
            model_inputs['adv_labels'] = adv_inputs['labels']
        else:
            model_inputs['adv_input_ids'] = normal_inputs['input_ids'].clone()
            model_inputs['adv_attention_mask'] = normal_inputs['attention_mask'].clone()
            model_inputs['adv_labels'] = normal_inputs['labels'].clone()
        
        return model_inputs


if __name__ == "__main__":
    from transformers import AutoTokenizer
    dataset = json.load(open("data/processed/STAIR-SFT_diverse.json", "r"))


    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
    tokenizer.pad_token = tokenizer.eos_token
    train_dataset = SafetyReasoningDataset(
        model_name="meta-llama/Llama-3.1-8B-Instruct",
        dataset=dataset,
        tokenizer=tokenizer,
        max_length=2048,
    )
    print(train_dataset.dataset[0])

    