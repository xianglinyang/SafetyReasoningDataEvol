"""
Assemble the dataset by evolving the original dataset.
The steps are:
1. Sample a question from the question pool.
2. Sample a question strategy.
3. Enrich the question template by the question strategy.
4. Generate the question by the enriched template.
5. Modify the answer with the answer reasoning.
6. Save the evolved question and answer.
"""
# TODO: assemble the question and answer into a jsonl file -> save in the data/evolved_data folder

import os
import json
import requests
import random
import logging
import torch
from tqdm import tqdm
import argparse
from datasets import load_dataset

from src.evol.questions import QuestionStrategy
from src.evol.answers import AnswerStrategy
from src.llm_zoo.utils import batch_invoke, load_model, load_tokenizer
from src.logger.config import setup_logging

logger = logging.getLogger(__name__)

#-------------------------Data evolution-------------------------
'''
Logic:
1. Load the question and answer strategies
2. Load the question and answer variants
3. Assemble the question and answer variants
4. Save the evolved question and answer
'''
class DataEvolver:
    def __init__(self):
        self.question_strategy = QuestionStrategy()
        self.answer_strategy = AnswerStrategy()

        # load the strategies that are supported by both question and answer strategies
        self._strategies = self.question_strategy._strategies
        logger.info(f"Strategies: {self._strategies}")
        self._load_question_templates()

    def _load_question_templates(self):
        self.question_templates = {}
        for strategy in self._strategies:
            self.question_templates[strategy] = self.question_strategy.templates[strategy]
        self.merged_question_templates = list()
        for strategy in self._strategies:
            self.merged_question_templates.extend(self.question_templates[strategy])

    def sample_strategy(self):
        strategy = random.choice(self._strategies)
        return strategy
    
    def evol_data(self, question_template, question: str, answer_output: str, refusal=True):
        question_instance = question_template.format(question=question)
        if refusal:
            answer_instance = self.answer_strategy.safe_cot(question, answer_output)
        else:
            answer_instance = self.answer_strategy.normal_cot(question, answer_output)
        return question_instance, answer_instance
  
    def evol_dataset(self, dataset: list, refusal=True) -> list:
        """Evolve an entire dataset of question-answer pairs"""

        if len(self.merged_question_templates) < len(dataset):
            logger.error(f"Not enough templates for the dataset. Expected at least {len(dataset)} templates, but only {len(self.merged_question_templates)} templates are available.")
            return []
        # shuffle the questions
        random.shuffle(self.merged_question_templates)

        evolved_dataset = list()
        for (question, output), question_template in zip(dataset, self.merged_question_templates):
            question_instance, answer_instance = self.evol_data(question_template, question, output, refusal)
            evolved_dataset.append({
                "evolved_question": question_instance.strip(),
                "evolved_answer": answer_instance.strip(),
            })
        return evolved_dataset


#-------------------------Data Loading and Saving-------------------------
# Adapt from https://raw.githubusercontent.com/GraySwanAI/circuit-breakers/refs/heads/main/data/circuit_breakers_val.json

PROCESSED_DATA_DIR = "data/processed"
RAW_DATA_DIR = "data/raw"

def download_file(url, file_path):
    response = requests.get(url)
    response.raise_for_status()

    dir = os.path.dirname(file_path)
    if not os.path.exists(dir):
        os.makedirs(dir)

    with open(file_path, "wb") as file:
        file.write(response.content)

def dump_json(dataset, file_path):
    dir = os.path.dirname(file_path)
    if not os.path.exists(dir):
        os.makedirs(dir)

    with open(file_path, "w") as file:
        json.dump(dataset, file, indent=4)

    logger.info(f"Dataset saved to {file_path}")


def download_circuitbreaker_dataset(train):
    logger.info(f"Downloading circuitbreaker {train} dataset...")
    if train:
        file = "https://raw.githubusercontent.com/GraySwanAI/circuit-breakers/refs/heads/main/data/circuit_breakers_train.json"
    else:
        file = "https://raw.githubusercontent.com/GraySwanAI/circuit-breakers/refs/heads/main/data/circuit_breakers_val.json"
    file_path = os.path.join(RAW_DATA_DIR, f'circuitbreaker_{"train" if train else "val"}.json')
    
    download_file(file, file_path)
    logger.info(f"Circuitbreaker {file_path} downloaded")


def download_dolly():
    logger.info(f"Downloading dolly dataset...")
    processed_file_path = os.path.join(PROCESSED_DATA_DIR, f'dolly.json')

    # dataset keys: category, prompt, llama3_output
    dataset = load_dataset("databricks/databricks-dolly-15k", split="train")

    evolved_dataset = []
    for data in dataset:
        context = data['context']
        instruction = data['instruction']
        category = data['category']
        response = data['response']
        
        question = context + instruction
        evolved_dataset.append({
            "category": category,
            "answer": response,
            "instruction": instruction,
            "context": context,
            "evolved_question": question,
        })

    dump_json(evolved_dataset, processed_file_path)
    logger.info(f"Dolly dataset saved to {processed_file_path}")

#-------------------------Data Processing-------------------------
def process_circuitbreaker_dataset(train):
    logger.info(f"Processing circuitbreaker {train} dataset...")
    raw_file_path = os.path.join(RAW_DATA_DIR, f'circuitbreaker_{"train" if train else "val"}.json')
    processed_file_path = os.path.join(PROCESSED_DATA_DIR, f'circuitbreaker_{"train" if train else "val"}.json')

    # dataset keys: category, prompt, llama3_output
    with open(raw_file_path, 'r') as f:
        dataset = json.load(f)

    data_evolver = DataEvolver()
    question_templates = data_evolver.merged_question_templates
    if len(question_templates) < len(dataset):
        logger.error(f"Not enough templates for the dataset. Expected at least {len(dataset)} templates, but only {len(question_templates)} templates are available.")
        return
    else:
        random.shuffle(question_templates)
        question_templates = question_templates[:len(dataset)]
    
    count = 0
    merged_dataset = []
    for data, question_template in tqdm(zip(dataset, question_templates), desc="Evolving dataset"):
        question = data['prompt']
        refusal_output = data['llama3_output']
        question_instance, answer_instance = data_evolver.evol_data(question_template, question, refusal_output, refusal=True)
        new_data = {
            "category": data['category'] if 'category' in data else "",
            "question": question,
            "refusal_output": refusal_output,
            "output": data["output"],
            "evolved_question": question_instance,
            "evolved_answer": answer_instance
        }
        merged_dataset.append(new_data)
        count += 1
        logger.info(f"Evolved {count} Question: {question_instance}")
        logger.info(f"Evolved {count} Answer: {answer_instance}")
    
    dump_json(merged_dataset, processed_file_path)
    logger.info(f"Evolved train dataset saved to {processed_file_path}")


def process_dolly():
    logger.info(f"Processing dolly dataset...")
    processed_file_path = os.path.join(PROCESSED_DATA_DIR, f'dolly.json')

    with open(processed_file_path, 'r') as f:
        dataset = json.load(f)
    
    data_evolver = DataEvolver()
    count = 0
    
    for data in tqdm(dataset, desc="Evolving Dolly dataset"):
        count += 1
        question = data['evolved_question']
        output = data['answer']
        evolved_answer = data_evolver.answer_strategy.normal_cot(question=question, response=output)
        data['evolved_answer'] = evolved_answer
        
        logger.info(f"{count} Question: {question}")
        logger.info(f"Evolved {count} Answer: {evolved_answer}")

    with open(processed_file_path, 'w') as f:
        json.dump(dataset, f)
    logger.info(f"Evolved dolly dataset saved to {processed_file_path}")


def process_instruction_following_dataset(model_name_or_path, dataset_path, device_map="cuda:0", batch_size=8, max_new_tokens=2048):
    """Process dataset and save with generated answers"""
    # Load llm
    logger.info(f"Loading model from {model_name_or_path}")
    
    # Load data
    with open(dataset_path, 'r') as f:
        dataset = json.load(f)
    
    # Extract questions
    questions = [data['evolved_question'] for data in dataset]
    
    # Generate answers
    logger.info(f"Generating answers for {len(questions)} questions")
    model = load_model(model_name_or_path, device_map=device_map)
    tokenizer = load_tokenizer(model_name_or_path)
    answers = batch_invoke(model, tokenizer, questions, batch_size=batch_size, max_new_tokens=max_new_tokens)

    # Update dataset with generated answers
    for data, answer in zip(dataset, answers):
        data[model_name_or_path] = answer
    
    # Save processed dataset
    with open(dataset_path, 'w') as f:
        json.dump(dataset, f)

    logger.info(f"Dataset processed and saved to {dataset_path}")



#TODO
# open instruction dataset
# open assistant dataset
# cot dataset
# flan v2 dataset

#-------------------------Main-------------------------
def main():
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--run_id", '-r', type=int)
    # args = parser.parse_args()
    # run_id = args.run_id

    # setup_logging(task_name="data_evol", run_id=run_id)
    setup_logging(task_name="data_evol", run_id=0)
    # logger.info("Assembling data...")

    # logger.info("Downloading circuitbreaker dataset...")
    # download_circuitbreaker_dataset(train=True)
    # download_circuitbreaker_dataset(train=False)

    # logger.info("Processing circuitbreaker dataset...")
    # process_circuitbreaker_dataset(train=True) 
    # process_circuitbreaker_dataset(train=False) 


    # logger.info("Processing dolly dataset...")
    # download_dolly()
    # process_dolly()
    # process_instruction_following_dataset(model_name_or_path="meta-llama/Llama-2-7b-chat-hf", dataset_path="data/processed/dolly.json", device_map="cuda:0", batch_size=4, max_new_tokens=2048)
    process_instruction_following_dataset(model_name_or_path="meta-llama/Llama-3.1-8B-Instruct", dataset_path="data/processed/dolly.json", device_map="cuda:0", batch_size=4, max_new_tokens=2048)
    # process_instruction_following_dataset(model_name_or_path="mistralai/Mistral-7B-Instruct-v0.2", dataset_path="data/processed/dolly.json", device_map="cuda:0", batch_size=4, max_new_tokens=2048)


if __name__ == "__main__":
    # import os
    # os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    # os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    
    main()