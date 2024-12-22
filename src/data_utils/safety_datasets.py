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
                ratio: float = 0.5,
                system_inst: str = None,
                ):
        super(SafetyReasoningDataset, self).__init__()
        self.model_name = model_name
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.ratio = ratio
        self.system_inst = system_inst
        self._load_data(dataset_name, split)
    

    def _load_data(self, dataset_name, split):
        # circuitbreaker original data+ processed data
        data_path = f"data/processed/{dataset_name}_{split}.json"
        with open(data_path, 'r') as f:
            circuitbreaker = json.load(f)
        # dolly dataset
        data_path = f"data/processed/dolly.json"
        with open(data_path, 'r') as f:
            dolly = json.load(f)
        
        self.dataset = list()
        # original data with evolved question and answer
        for data in circuitbreaker:
            evolved_question = data['evolved_question']
            evolved_answer = data['evolved_answer']
            self.dataset.append({
                "question": evolved_question,
                "answer": evolved_answer
            })
        # original data with original question and answer
        for data in circuitbreaker:
            question = data['question']
            evolved_answer = data['evolved_answer']
            self.dataset.append({
                "question": question,
                "answer": evolved_answer
            })
        # load other instructions data
        # dolly dataset
        # TODO use the llm generated data
        num = int(len(self.dataset) / self.ratio - len(self.dataset))
        dolly = random.sample(dolly, num)

        for data in dolly:
            question = data['evolved_question']
            answer = data['evolved_answer']
            refusal_part = answer.split('#### Response')[0]
            output = data[self.model_name]
            new_answer = refusal_part+"#### Response\n"+output
            self.dataset.append({
                "question": question,
                "answer": new_answer
            })
        
        # shuffle the dataset
        random.shuffle(self.dataset)

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        example = self.dataset[i]
        
        question = example['question']
        answer = example['answer']

        # format data
        messages = list()
        # if self.system_inst is not None:
        #     messages.append({"role": "system", "content": self.system_inst})
        messages.append({"role": "user", "content": question})
        messages.append({"role": "assistant", "content": answer})

        # encode the messages
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
        # Create labels (same as input_ids but with -100 for non-assistant tokens)
        labels = encodings['input_ids'][0].clone()

        # assistant_tokens_mask
        formatted_text_wo_assistant = self.tokenizer.apply_chat_template(messages[:-1], tokenize=False)
        encodings_wo_assistant = self.tokenizer(formatted_text_wo_assistant, 
                                                return_tensors='pt', 
                                                padding=False)

        # mask the prompt part for avoiding loss
        labels[:encodings_wo_assistant.input_ids.shape[1]] = -100    
    
        model_inputs['labels'] = labels
        return model_inputs
    

    