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
        self._strategies = list(set(self.question_strategy._strategies) & set(self.answer_strategy._strategies))
        logger.info(f"Strategies: {self._strategies}")

        self.templates = self._load_templates()

    def _load_templates(self):
        self.templates = {}
        for strategy in self._strategies:
            question_templates = self.question_strategy.templates[strategy]
            answer_templates = self.answer_strategy.templates[strategy]
            min_len = min(len(question_templates), len(answer_templates))
            question_templates = random.sample(question_templates, min_len)
            answer_templates = random.sample(answer_templates, min_len)
            # zip the question and answer templates
            self.templates[strategy] = list(zip(question_templates, answer_templates))
        return self.templates
    
    def sample_strategy(self):
        strategy = random.choice(self._strategies)
        return strategy
    
    def evol_single_data(self, question: str, question_category="harmful and toxic content"):
        strategy = self.sample_strategy()
        question_template, answer_template = random.choice(self.templates[strategy])
        question_instance = question_template.format(question=question)
        answer_instance = answer_template.format(question=question, question_category=question_category)
        return question_instance, answer_instance
    
    def evol_data(self, question_template, answer_template, question: str, question_category="harmful and toxic content"):
        question_instance = question_template.format(question=question)
        answer_instance = answer_template.format(question=question, question_category=question_category)
        return question_instance, answer_instance
    
    def evol_dataset(self, dataset: list) -> list:
        """Evolve an entire dataset of question-answer pairs"""
        all_qa_pairs = list()
        for strategy in self._strategies:
            all_qa_pairs.extend(self.templates[strategy])
        if len(all_qa_pairs) < len(dataset):
            logger.error(f"Not enough templates for the dataset. Expected at least {len(dataset)} templates, but only {len(all_qa_pairs)} templates are available.")
            return []
        
        # shuffle the pairs
        random.shuffle(all_qa_pairs)

        evolved_dataset = list()
        for (question, question_category), (question_template, answer_template) in zip(dataset, all_qa_pairs):
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
    if train:
        file = "https://raw.githubusercontent.com/GraySwanAI/circuit-breakers/refs/heads/main/data/circuit_breakers_train.json"
    else:
        file = "https://raw.githubusercontent.com/GraySwanAI/circuit-breakers/refs/heads/main/data/circuit_breakers_val.json"
    file_path = os.path.join(RAW_DATA_DIR, f'circuitbreaker_{"train" if train else "val"}.json')
    
    download_file(file, file_path)
    logger.info(f"Circuitbreaker {file_path} downloaded")


def process_circuitbreaker_dataset(train):
    raw_file_path = os.path.join(RAW_DATA_DIR, f'circuitbreaker_{"train" if train else "val"}.json')
    processed_file_path = os.path.join(PROCESSED_DATA_DIR, f'circuitbreaker_{"train" if train else "val"}.json')

    # dataset keys: category, prompt, llama3_output
    with open(raw_file_path, 'r') as f:
        dataset = json.load(f)

    question_category_pairs = [(dataset[i]['prompt'], dataset[i]['category']) for i in range(len(dataset))]

    data_evolver = DataEvolver()
    evolved_dataset = data_evolver.evol_dataset(question_category_pairs)

    # merge the original dataset and the evolved dataset
    merged_dataset = []
    for i in range(len(dataset)):
        question = dataset[i]['prompt']
        question_category = "harmful and toxic content" if 'category' not in dataset[i] else dataset[i]['category']
        refusal_answer = dataset[i]['llama3_output']
        output = dataset[i]['output']
        evolved_question, evolved_answer = evolved_dataset[i]['evolved_question'], evolved_dataset[i]['evolved_answer']
        
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

#-------------------------Main-------------------------
def main():
    setup_logging(task_name="data_evol")
    logger.info("Assembling data...")

    logger.info("Downloading circuitbreaker dataset...")
    # download_circuitbreaker_dataset(train=True)
    # download_circuitbreaker_dataset(train=False)

    logger.info("Processing circuitbreaker dataset...")
    process_circuitbreaker_dataset(train=True) 
    # process_circuitbreaker_dataset(train=False) 


if __name__ == "__main__":
    main()
