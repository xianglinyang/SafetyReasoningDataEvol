import json
import random
from typing import Dict, List

import torch
import transformers
from torch.utils.data import Dataset

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
        # circuitbreaker original data+ processed data
        data_path = f"data/processed/{dataset_name}_{split}_metadata_final.json"
        with open(data_path, 'r') as f:
            circuitbreaker = json.load(f)
        if split == 'val':
            circuitbreaker = circuitbreaker[:100]
    
        self.refusal_dataset = list()
        # original data with evolved question and answer
        for data in circuitbreaker:
            question_variants = data['evolved_variants']
            question = data['prompt']
            answer = data['evolved_answer_modified']
            refusal = data['llama3_output']
            if self.include_reasoning:
                answer = self._simple_answer(answer, refusal)
            else:
                # answer = "I cannot help with that."
                answer = refusal

            self.refusal_dataset.append({
                "question": question,
                "answer": answer
            })

            if self.include_variants:   
                # Randomly select one variant from question_variants if available
                variant_questions = [q for _, q in question_variants.items()]
                question = random.choice(variant_questions)
                self.refusal_dataset.append({
                    "question": question,
                    "answer": answer
                })
        random.shuffle(self.refusal_dataset)
        print("refusal_dataset length:", len(self.refusal_dataset))
            
        # ==========================  Retain ========================== #
        # dolly dataset
        data_path = f"data/processed/dolly_{split}_metadata_final.json"
        with open(data_path, 'r') as f:
            dolly = json.load(f)
        if split == 'val':
            dolly = dolly[:100]
        
        self.retain_dataset = list()
        for data in dolly:
            question = data['evolved_question']
            answer = data['evolved_answer_modified']

            # clean format
            output = data[self.model_name]
            if output.startswith('assistant\n\n'):
                output = output[10:]
            
            if self.include_reasoning:
                refusal_part = answer.split('#### Response')[0]
                new_answer = refusal_part+"#### Response\n"+output
            else:
                new_answer = output

            self.retain_dataset.append({
                "question": question,
                "answer": new_answer
            })

        # shuffle the dataset
        random.shuffle(self.retain_dataset)
        print("retain_dataset length:", len(self.retain_dataset))
    
    def _simple_answer(self, answer, refusal=None):
        reasoning_part = answer.split('#### Response')[0]
        if refusal:
            simple_answer = f"#### Response\n{refusal}"
        else:
            simple_answer = "#### Response\nI cannot help with that."
        return reasoning_part + simple_answer
    
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

        # retain example
        retain_example = self.retain_dataset[i % len(self.retain_dataset)]
        retain_question = retain_example['question']
        retain_answer = retain_example['answer']
        retain_inputs = self._format_data(retain_question, retain_answer)

        model_inputs = dict(
            input_ids=refusal_inputs['input_ids'],
            attention_mask=refusal_inputs['attention_mask'],
            labels=refusal_inputs['labels'],
            retain_input_ids=retain_inputs['input_ids'],
            retain_attention_mask=retain_inputs['attention_mask'],
            retain_labels=retain_inputs['labels']
        )
        return model_inputs


if __name__ == "__main__":
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
    tokenizer.pad_token = tokenizer.eos_token
    train_dataset = SafetyReasoningDataset(
        model_name="meta-llama/Llama-3.1-8B-Instruct",
        dataset_name="circuitbreaker",
        split="train",
        tokenizer=tokenizer,
        max_length=2048,
        include_variants=True,
        include_reasoning=True
    )
    print(train_dataset.refusal_dataset[0])
    print(train_dataset.retain_dataset[0])

    