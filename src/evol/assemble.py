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

from src.evol.question_evol import QuestionEvol
from src.evol.answers import AnswerEvol
from src.llm_zoo.utils import batch_invoke, load_model, load_tokenizer
from src.llm_zoo.api_base_models import BaseLLM
from src.llm_zoo.api_base_models import OpenAILLM
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
        self.question_evol = QuestionEvol()
        self.answer_evol = AnswerEvol()
    
    def evol_question(self, question_llm: BaseLLM, question):
        question_variants = self.question_evol.generate_prompt_variants(question, question_llm)
        return question_variants
    
    def evol_answer(self, answer_llm: BaseLLM, question, question_type, answer_output):
        answer_block, metadata = self.answer_evol.generate_evol_answer(answer_llm, question, question_type, answer_output, return_metadata=True)
        return answer_block, metadata
    
    def evol_data(self, question_llm: BaseLLM, answer_llm: BaseLLM, question, answer_output, question_type):
        question_variants = self.question_evol.generate_prompt_variants(question, question_llm)
        answer_block, metadata = self.answer_evol.generate_evol_answer(answer_llm, question, question_type, answer_output, return_metadata=True)
        return question_variants, answer_block, metadata

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
def process_circuitbreaker_train_dataset():
    logger.info(f"Processing circuitbreaker train dataset...")
    raw_file_path = os.path.join(RAW_DATA_DIR, f'circuitbreaker_train.json')
    processed_file_path = os.path.join(PROCESSED_DATA_DIR, f'circuitbreaker_train.json')

    question_llm = OpenAILLM(model_name="gpt-4o-mini")
    answer_llm = OpenAILLM(model_name="gpt-4.1-mini")

    # dataset keys: category, prompt, llama3_output
    with open(raw_file_path, 'r') as f:
        dataset = json.load(f)

    data_evolver = DataEvolver()
    new_dataset = []
    count = 0

    for data in tqdm(dataset, desc="Evolving dataset"):
        question = data['prompt']
        question_variants, answer_instance, metadata = data_evolver.evol_data(question_llm, answer_llm, question, answer=None, question_type="harmful")

        data['evolved_variants'] = question_variants
        data['evolved_answer'] = answer_instance
        data['metadata'] = metadata
        count += 1
        new_dataset.append(data)
        logger.info(f"Evolved {count} Question Variants: {question_variants}")
        logger.info(f"Evolved {count} Answer: {answer_instance}")
    
    dump_json(new_dataset, processed_file_path)
    logger.info(f"Evolved train dataset saved to {processed_file_path}")


def process_circuitbreaker_val_dataset():
    logger.info(f"Processing circuitbreaker val dataset...")
    raw_file_path = os.path.join(RAW_DATA_DIR, f'circuitbreaker_val.json')
    processed_file_path = os.path.join(PROCESSED_DATA_DIR, f'circuitbreaker_val.json')

    question_llm = OpenAILLM(model_name="gpt-4o-mini")
    answer_llm = OpenAILLM(model_name="gpt-4.1-mini")

    # dataset keys: category, prompt, llama3_output
    with open(raw_file_path, 'r') as f:
        dataset = json.load(f)

    data_evolver = DataEvolver()
    new_dataset = []
    count = 0
    for data in tqdm(dataset, desc="Evolving dataset"):
        question = data['prompt']
        output = data['llama3_output']
        question_variants, answer_instance, metadata = data_evolver.evol_data(question_llm, answer_llm, question, answer=output, question_type="benign")

        data['evolved_variants'] = question_variants
        data['evolved_answer'] = answer_instance
        data['metadata'] = metadata
        count += 1
        logger.info(f"Evolved {count} Question Variants: {question_variants}")
        logger.info(f"Evolved {count} Answer: {answer_instance}")
        new_dataset.append(data)
    
    dump_json(new_dataset, processed_file_path)
    logger.info(f"Evolved train dataset saved to {processed_file_path}")


def process_dolly():
    logger.info(f"Processing dolly dataset...")
    processed_file_path = os.path.join(PROCESSED_DATA_DIR, f'dolly.json')

    answer_llm = OpenAILLM(model_name="gpt-4.1-mini")

    with open(processed_file_path, 'r') as f:
        dataset = json.load(f)
    
    data_evolver = DataEvolver()
    new_dataset = []
    count = 0
    
    for data in tqdm(dataset, desc="Evolving Dolly dataset"):
        count += 1
        question = data['evolved_question']
        output = data['answer']

        answer_instance, metadata = data_evolver.evol_answer(answer_llm, question, "benign", output, return_metadata=True)
        data['evolved_answer'] = answer_instance
        data['metadata'] = metadata
        new_dataset.append(data)
        
        logger.info(f"{count} Question: {question}")
        logger.info(f"Evolved {count} Answer: {answer_instance}")
    
    # randomly split it into train and val with ratio=8:2
    random.shuffle(new_dataset)
    train_dataset = new_dataset[:int(len(new_dataset) * 0.8)]
    val_dataset = new_dataset[int(len(new_dataset) * 0.8):]

    dump_json(train_dataset, os.path.join(PROCESSED_DATA_DIR, f'dolly_train.json'))
    dump_json(val_dataset, os.path.join(PROCESSED_DATA_DIR, f'dolly_val.json'))
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
    dump_json(dataset, dataset_path)

    logger.info(f"Dataset processed and saved to {dataset_path}")


# TODO
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
    # logger.info("Assembling data...")

    # logger.info("Downloading circuitbreaker dataset...")
    # download_circuitbreaker_dataset(train=True)
    # download_circuitbreaker_dataset(train=False)

    # logger.info("Processing circuitbreaker dataset...")
    # process_circuitbreaker_train_dataset() 
    # process_circuitbreaker_val_dataset() 

    # logger.info("Processing dolly dataset...")
    # download_dolly()
    # process_dolly()

    # logger.info("Processing instruction following dataset...")
    # logger.info("Processing Llama-3.1-8B-Instruct dataset...")
    # process_instruction_following_dataset(model_name_or_path="meta-llama/Llama-3.1-8B-Instruct", dataset_path="data/processed/dolly_train.json", device_map="cuda:0", batch_size=4, max_new_tokens=2048)
    # process_instruction_following_dataset(model_name_or_path="meta-llama/Llama-3.1-8B-Instruct", dataset_path="data/processed/dolly_val.json", device_map="cuda:0", batch_size=4, max_new_tokens=2048)
    
    # logger.info("Processing mistral dataset...")
    # process_instruction_following_dataset(model_name_or_path="mistralai/Mistral-7B-Instruct-v0.2", dataset_path="data/processed/dolly_val.json", device_map="cuda:0", batch_size=4, max_new_tokens=2048)
    # process_instruction_following_dataset(model_name_or_path="mistralai/Mistral-7B-Instruct-v0.2", dataset_path="data/processed/dolly_train.json", device_map="cuda:0", batch_size=4, max_new_tokens=2048)

    logger.info("Processing Qwen dataset...")
    process_instruction_following_dataset(model_name_or_path="Qwen/Qwen2.5-7B-Instruct", dataset_path="data/processed/dolly_train.json", device_map="cuda:0", batch_size=4, max_new_tokens=2048)
    process_instruction_following_dataset(model_name_or_path="Qwen/Qwen2.5-7B-Instruct", dataset_path="data/processed/dolly_val.json", device_map="cuda:0", batch_size=4, max_new_tokens=2048)


if __name__ == "__main__":    
    main()