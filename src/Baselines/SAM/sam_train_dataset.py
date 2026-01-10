from torch.utils.data import Dataset
from datasets import load_dataset
import torch
import numpy as np
import json
import random
import csv

random.seed(0)

def data_reader(dataset_name: str, prob=1.0):
    '''
    prob=1.0: one shot prompt
    prob=0.5: no prompt
    '''
    if dataset_name == "ultrachat":
        # ======================= Retain ======================= #
        ds = load_dataset("HuggingFaceH4/ultrachat_200k", split="test_sft")
        orig_s = []
        for example in ds:
            messages = example["messages"]
            if len(messages) < 2: 
                continue
            else:
                probability = np.random.random()
                if probability <= prob:
                   orig_s.append(messages)
                else:
                    message = messages[:2]
                    message[0]['content'] = ""
                    orig_s.append(message)
            if len(orig_s) >= 2000:
                break

        random.shuffle(orig_s)
        return orig_s
    
    elif dataset_name == "xstest":
        # from https://github.com/paul-rottger/exaggerated-safety
        with open(f'data/processed/xstest_v2_completions_gpt4_gpteval.csv', newline='') as f:
            data = [dict(row) for row in csv.DictReader(f)]
            data = [row for row in data if row['final_label'] == "1_full_compliance"]
        
        borderline_orig_s = []
        for i, d in enumerate(data * 50):
            prompt = d['prompt']
            completion = d['completion']
            
            switch = np.random.random()
            if switch <= prob:
                messages = [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": completion}
                ]
            else:
                messages = [
                    {"role": "user", "content": ""},
                    {"role": "assistant", "content": completion}
                ]

            borderline_orig_s.append(messages)
        random.shuffle(borderline_orig_s)
        return borderline_orig_s
    
    elif dataset_name == "circuitbreaker-train-retain":
        with open("data/raw/circuit_breakers_train.json") as file:
            dataset = json.load(file)
        random.shuffle(dataset)
        dataset = dataset[:2000]
        
        refusal_retain_orig = []
        for i, d in enumerate(dataset * 2):
            prompt = d['prompt']
            completion = d['llama3_output']
            switch = np.random.random()
            if switch <= prob:
                messages = [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": completion}
                ]
            else:
                messages = [
                    {"role": "user", "content": ""},
                    {"role": "assistant", "content": completion}
                ]
            refusal_retain_orig.append(messages)
        random.shuffle(refusal_retain_orig)
        return refusal_retain_orig
    
    elif dataset_name == "circuitbreaker-train-cb":
        with open("data/raw/circuit_breakers_train.json") as file:
            dataset = json.load(file)
        res = []
        for d in dataset:
            prompt = d['prompt']
            completion = d['output']
            switch = np.random.random()
            if switch <= prob:
                messages = [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": completion}
                ]
            else:
                messages = [
                    {"role": "user", "content": ""},
                    {"role": "assistant", "content": completion}
                ]
            res.append(messages)
        random.shuffle(res)
        return res
    
    elif dataset_name == "circuitbreaker-val":
        with open("data/raw/circuit_breakers_val.json") as file:
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

class SAMDataset(Dataset):
    def __init__(self, dataset, tokenizer, max_length=1024):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.dataset)
    
    def _format_data(self, messages):
        full_text = self.tokenizer.apply_chat_template(messages, tokenize=False)
        enc = self.tokenizer(
            full_text,
            max_length=self.max_length,
            truncation=True,
            padding=False,
            return_tensors=None,
        )

        # 只到 user 为止（用于确定 prompt 长度）
        prompt_text = self.tokenizer.apply_chat_template(messages[:-1], tokenize=False)
        prompt_enc = self.tokenizer(
            prompt_text,
            max_length=self.max_length,
            truncation=True,
            padding=False,
            return_tensors=None,
        )
        prompt_len = len(prompt_enc["input_ids"])

        input_ids = torch.tensor(enc["input_ids"], dtype=torch.long)
        attention_mask = torch.tensor(enc["attention_mask"], dtype=torch.long)

        labels = input_ids.clone()
        labels[:prompt_len] = -100  # mask prompt

        return dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )

    def __getitem__(self, i):
        messages = self.dataset[i]
        model_inputs = self._format_data(messages)
        return model_inputs

def sam_collate_fn(batch):
    """
    Collate function for SAM dataset batching.
    Pads sequences to the same length within the batch.
    
    Args:
        batch: List of dicts from SAMDataset.__getitem__
            Each dict contains 'input_ids', 'attention_mask', 'labels'
    
    Returns:
        Dict with batched and padded tensors
    """
    # Find max length in this batch
    max_len = max(item['input_ids'].size(0) for item in batch)
    
    batch_size = len(batch)
    
    # Initialize padded tensors
    # Use 0 for padding (will be masked by attention_mask anyway)
    input_ids = torch.zeros((batch_size, max_len), dtype=torch.long)
    attention_mask = torch.zeros((batch_size, max_len), dtype=torch.long)
    labels = torch.full((batch_size, max_len), -100, dtype=torch.long)
    
    # Fill in the actual values
    for i, item in enumerate(batch):
        seq_len = item['input_ids'].size(0)
        input_ids[i, :seq_len] = item['input_ids']
        attention_mask[i, :seq_len] = item['attention_mask']
        labels[i, :seq_len] = item['labels']
    
    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels,
    }
