from torch.utils.data import Dataset
from datasets import load_dataset
import transformers
from typing import Dict
import torch
import numpy as np
from tqdm import tqdm
import json
import random
import csv

random.seed(0)

def data_reader(dataset_name: str):
    if dataset_name == "ultrachat":
        # ======================= Retain ======================= #
        ds = load_dataset("HuggingFaceH4/ultrachat_200k", split="test_sft")
        orig_s = []
        for example in ds:
            messages = example["messages"]
            if len(messages) < 2: 
                continue
            else:
                orig_s.append(messages)
        return orig_s
    elif dataset_name == "xstest":
        # from https://github.com/paul-rottger/exaggerated-safety
        with open(f'data/processed/xstest_v2_completions_gpt4_gpteval.csv', newline='') as f:
            data = [dict(row) for row in csv.DictReader(f)]
            data = [row for row in data if row['final_label'] == "1_full_compliance"]
        borderline_orig_s = []
        for d in data:
            prompt = d['prompt']
            completion = d['completion']
            messages = [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": completion}
            ]
            borderline_orig_s.append(messages)
        return borderline_orig_s
    
    elif dataset_name == "circuitbreaker-train-retain":

        with open("data/processed/circuitbreaker_train.json") as file:
            dataset = json.load(file)
        dataset = dataset[:2000]
        borderline_orig_s = []
        for d in dataset:
            prompt = d['prompt']
            completion = d['llama3_output']
            messages = [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": completion}
            ]
            borderline_orig_s.append(messages)
        return borderline_orig_s
    
    elif dataset_name == "circuitbreaker-train-cb":
        with open("data/processed/circuitbreaker_train.json") as file:
            dataset = json.load(file)
        res = []
        for d in dataset:
            prompt = d['prompt']
            completion = d['output']
            messages = [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": completion}
            ]
            res.append(messages)
        return res
    elif dataset_name == "circuitbreaker-val":
        with open("data/processed/circuitbreaker_val.json") as file:
            dataset = json.load(file)
        res = []
        for d in dataset:
            prompt = d['prompt']
            completion = d['output']
            messages = [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": completion}
            ]
            res.append(messages)
        return res


class CircuitBreakerDataset(Dataset):
    
    def __init__(self, 
        refusal_dataset,
        retain_dataset,
        tokenizer: transformers.PreTrainedTokenizer, 
        model_name_or_path,
        max_length = 1024
    ):
        super(CircuitBreakerDataset, self).__init__()

        self.model_name_or_path = model_name_or_path.lower()
        self.max_length = max_length
        self.tokenizer = tokenizer
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        self.refusal_dataset = refusal_dataset
        self.retain_dataset = retain_dataset
    
    def _format_data(self, messages):
        
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
        return min(len(self.refusal_dataset), len(self.retain_dataset))
    
    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        refusal_example = self.refusal_dataset[i]
        retain_example = self.retain_dataset[i]

        refusal_inputs = self._format_data(refusal_example)
        retain_inputs = self._format_data(retain_example)

        model_inputs = dict(
            input_ids_refusal=refusal_inputs['input_ids'],
            attention_mask_refusal=refusal_inputs['attention_mask'],
            input_ids_retain=retain_inputs['input_ids'],
            attention_mask_retain=retain_inputs['attention_mask'],
        )
        return model_inputs

if __name__ == "__main__":
    from transformers import AutoTokenizer
    dataset_name = "circuitbreaker-val"
    refusal_dataset = data_reader(dataset_name)
    retain_dataset = data_reader(dataset_name)
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
    model_name_or_path = "meta-llama/Llama-3.1-8B-Instruct"
    max_length = 1024
    dataset = CircuitBreakerDataset(refusal_dataset, retain_dataset, tokenizer, model_name_or_path, max_length)
    print(dataset[0])