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
        # dataset = dataset[:2000]
        
        refusal_retain_orig = []
        for i, d in enumerate(dataset):
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
    
    def _format_refusal_data(self, messages):
        cb_tokenized_kwargs = dict(max_length=512, padding='max_length', truncation=True, return_tensors="pt")

        formatted_text_full = self.tokenizer.apply_chat_template(messages, tokenize=False)
        formatted_prompt_partial = self.tokenizer.apply_chat_template(messages[:-1], tokenize=False, add_generation_prompt=True)

        cb_request, cb_response = formatted_text_full[:len(formatted_prompt_partial)], formatted_text_full[len(formatted_prompt_partial):]
        
        self.tokenizer.padding_side = "left"
        tokenized_request_circuit_breaker = self.tokenizer(cb_request, **cb_tokenized_kwargs)
        self.tokenizer.padding_side = "right"
        response_tokenized_circuit_breaker = self.tokenizer(cb_response, add_special_tokens=False, **cb_tokenized_kwargs)
        self.tokenizer.padding_side = "left"

        combined_input_ids_circuit_breaker = torch.cat([tokenized_request_circuit_breaker["input_ids"], response_tokenized_circuit_breaker["input_ids"]], dim=1)
        combined_attention_mask_circuit_breaker = torch.cat([tokenized_request_circuit_breaker["attention_mask"], response_tokenized_circuit_breaker["attention_mask"]], dim=1)

        return dict(
            input_ids=combined_input_ids_circuit_breaker.squeeze(0),
            attention_mask=combined_attention_mask_circuit_breaker.squeeze(0),
        )
    
    def _format_retain_data(self, messages):
        tokenize_kwargs = dict(max_length=1024, padding="max_length", truncation=True, return_tensors="pt")

        orig_s_retain = self.tokenizer.apply_chat_template(messages, tokenize=False)
        tokenized_inputs_retain = self.tokenizer(orig_s_retain, **tokenize_kwargs)

        return dict(
            input_ids=tokenized_inputs_retain["input_ids"].squeeze(0),
            attention_mask=tokenized_inputs_retain["attention_mask"].squeeze(0),
        )

    def __len__(self):
        return min(len(self.refusal_dataset), len(self.retain_dataset))
    
    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        refusal_example = self.refusal_dataset[i]
        retain_example = self.retain_dataset[i]

        refusal_inputs = self._format_refusal_data(refusal_example)
        retain_inputs = self._format_retain_data(retain_example)

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