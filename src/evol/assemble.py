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
import argparse
from datasets import load_dataset

from src.evol.questions import QuestionStrategy
from src.evol.answers import AnswerStrategy
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
        self.question_templates = self._load_question_templates()

    def _load_question_templates(self):
        self.question_templates = {}
        for strategy in self._strategies:
            self.question_templates[strategy] = self.question_strategy.templates[strategy]
        return self.question_templates

    def sample_strategy(self):
        strategy = random.choice(self._strategies)
        return strategy
    
    def evol_data(self, question_template, answer_template, question: str, question_category: str, answer_output: str):
        question_instance = question_template.format(question=question)
        answer_instance = answer_template.format(category=question_category, output=answer_output)
        return question_instance, answer_instance
    
    def generate_templates(self, answer_output="refusal"):
        # load the question templates
        question_templates = list()
        for strategy in self._strategies:
            question_templates.extend(self.question_templates[strategy])
        # load the answer template
        if answer_output == "refusal":
            answer_template = self.answer_strategy.refusal_format
        elif answer_output == "normal":
            answer_template = self.answer_strategy.normal_format
        else:
            logger.error(f"Invalid output type: {answer_output}")
            return None, None
        return question_templates, answer_template
  
    def evol_dataset(self, dataset: list, output="refusal") -> list:
        """Evolve an entire dataset of question-answer pairs"""
        question_templates, answer_template = self.generate_templates(output)
        if len(question_templates) < len(dataset):
            logger.error(f"Not enough templates for the dataset. Expected at least {len(dataset)} templates, but only {len(question_templates)} templates are available.")
            return []
        # shuffle the questions
        random.shuffle(question_templates)

        evolved_dataset = list()
        for (question, question_category, output), question_template in zip(dataset, question_templates):
            if question_category is None:
                question_category = "unsafe content"
            question_instance, answer_instance = self.evol_data(question_template, answer_template, question, question_category)
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


def process_circuitbreaker_dataset(train):
    logger.info(f"Processing circuitbreaker {train} dataset...")
    raw_file_path = os.path.join(RAW_DATA_DIR, f'circuitbreaker_{"train" if train else "val"}.json')
    processed_file_path = os.path.join(PROCESSED_DATA_DIR, f'circuitbreaker_{"train" if train else "val"}.json')

    # dataset keys: category, prompt, llama3_output
    with open(raw_file_path, 'r') as f:
        dataset = json.load(f)
    data_evolver = DataEvolver()

    question_templates, answer_template = data_evolver.generate_templates(answer_output="refusal")
    if len(question_templates) < len(dataset):
        logger.error(f"Not enough templates for the dataset. Expected at least {len(dataset)} templates, but only {len(question_templates)} templates are available.")
        return
    else:
        random.shuffle(question_templates)
        question_templates = question_templates[:len(dataset)]

    # merge the original dataset and the evolved dataset
    merged_dataset = []
    for i in range(len(dataset)):
        question_template = question_templates[i]
        question = dataset[i]['prompt']
        question_category = "unsafe content" if 'category' not in dataset[i] else dataset[i]['category']
        refusal_answer = dataset[i]['llama3_output']
        output = dataset[i]['output']
        evolved_question = question_template.format(question=question)
        evolved_answer = answer_template.format(category=question_category, output=refusal_answer)
        
        merged_dataset.append({
            "category": question_category,
            "refusal_answer": refusal_answer,
            "output": output,
            "question": question,
            "evolved_question": evolved_question,
            "evolved_answer": evolved_answer
        })

    dump_json(merged_dataset, processed_file_path)
    logger.info(f"Evolved train dataset saved to {processed_file_path}")


def process_dolly():
    logger.info(f"Processing dolly dataset...")
    processed_file_path = os.path.join(PROCESSED_DATA_DIR, f'dolly.json')

    # dataset keys: category, prompt, llama3_output
    dataset = load_dataset("databricks/databricks-dolly-15k", split="train")
    
    data_evolver = DataEvolver()
    _, answer_template = data_evolver.generate_templates(answer_output="normal")

    evolved_dataset = []
    for data in dataset:
        context = data['context']
        instruction = data['instruction']
        category = data['category']
        response = data['response']
        
        question = context + instruction
        evolved_answer = answer_template.format(category=category, output=response)
        evolved_dataset.append({
            "category": category,
            "answer": response,
            "evolved_question": question,
            "evolved_answer": evolved_answer
        })

    dump_json(evolved_dataset, processed_file_path)
    logger.info(f"Evolved dolly dataset saved to {processed_file_path}")

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
    # logger.info("Assembling data...")

    # logger.info("Downloading circuitbreaker dataset...")
    # download_circuitbreaker_dataset(train=True)
    # download_circuitbreaker_dataset(train=False)

    # logger.info("Processing circuitbreaker dataset...")
    process_circuitbreaker_dataset(train=True) 
    process_circuitbreaker_dataset(train=False) 

    # logger.info("Processing dolly dataset...")
    process_dolly()


if __name__ == "__main__":
    main()
