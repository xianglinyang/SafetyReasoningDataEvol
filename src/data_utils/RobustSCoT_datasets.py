import json
import random
from typing import Dict

import torch
import transformers
from torch.utils.data import Dataset
import numpy as np

'''Dataset for Safety Reasoning'''
class SafetyReasoningDataset(Dataset):
    
    def __init__(self, 
                dataset_name: str,      # 'circuitbreaker'
                split: str,             # train/val/test
                model_name: str,         # 'llama3'
                tokenizer: transformers.PreTrainedTokenizer, 
                max_length: int = 2048,
                include_variants=True,
                include_reasoning=True
                ):
        super(SafetyReasoningDataset, self).__init__()
        self.model_name = model_name
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.include_variants = include_variants
        self.include_reasoning = include_reasoning
        self._load_data(dataset_name, split)

    def _load_data(self, dataset_name, split):
        # ======================= Circuitbreaker ======================= #
        data_path = f"data/processed/STAIR-SFT_diverse.json"

        with open(data_path, 'r') as f:
            dataset = json.load(f)
        length = len(dataset)
        if split == 'train':
            dataset = dataset[:int(length*0.995)]
        elif split == 'val':
            dataset = dataset[int(length*0.995):]
        
        self.refusal_dataset = list()
        # original data with evolved question and answer
        for data in dataset:
            answer = data['cot']+ "\n"+data['answer']
            source = data['source']
            # if source == 'PKU-SafeRLHF':
            #     prob = np.random.rand()
            #     if prob < 0.5:
            #         question = data['instruction']
            #     else:
            #         try:
            #             mutations = data['mutations']
            #             question = random.choice(mutations)['text']
            #         except:
            #             question = data['instruction']
            # else:
            #     question = data['instruction']
            question = data['instruction']

            self.refusal_dataset.append({
                "question": question,
                "answer": answer
            })

        random.shuffle(self.refusal_dataset)
        print("refusal_dataset length:", len(self.refusal_dataset))
    

    
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
        return len(self.refusal_dataset)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        # Get items from both datasets
        refusal_example = self.refusal_dataset[i % len(self.refusal_dataset)]  
        refusal_question = refusal_example['question']
        refusal_answer = refusal_example['answer']
        refusal_inputs = self._format_data(refusal_question, refusal_answer)
        model_inputs = dict(
            input_ids=refusal_inputs['input_ids'],
            attention_mask=refusal_inputs['attention_mask'],
            labels=refusal_inputs['labels']
        )
        return model_inputs


if __name__ == "__main__":
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
    tokenizer.pad_token = tokenizer.eos_token
    train_dataset = SafetyReasoningDataset(
        model_name="meta-llama/Llama-3.1-8B-Instruct",
        dataset_name="STAIR-SFT_diverse",
        split="train",
        tokenizer=tokenizer,
        max_length=2048,
        include_variants=True,
        include_reasoning=True
    )
    print(train_dataset.refusal_dataset[0])

    