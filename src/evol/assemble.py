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

from src.evol.questions import QuestionStrategy
from src.evol.answers import AnswerStrategy

#-------------------------Data evolution-------------------------
class DataEvolver:
    def __init__(self):
        self.question_strategy = QuestionStrategy()
        self.answer_strategy = AnswerStrategy()

    def evol_question(self, question: str) -> str:
        """Evolve a single question using sampled strategy"""
        question_instance = self.question_strategy.enrich_question(question)
        print("Question instance: ", question_instance)
        return question_instance
    
    def evol_answer(self, question: str, question_category=None) -> str:
        """Evolve an answer using the same strategy as the question"""
        answer_instance = self.answer_strategy.enrich_answer(question, question_category)
        print("Answer instance: ", answer_instance)
        return answer_instance

    def evol_data(self, question: str, question_category=None) -> tuple:
        """Evolve a single question-answer pair"""
        question_instance = self.evol_question(question)
        answer_instance = self.evol_answer(question, question_category)
        return question_instance, answer_instance
    
    def evol_dataset(self, dataset: list) -> list:
        """Evolve an entire dataset of question-answer pairs"""
        evolved_dataset = []
        for question, question_category in dataset:
            question_instance, answer_instance = self.evol_data(question, question_category)
            evolved_dataset.append({
                "question": question_instance.strip(),
                "category": question_category,
                "answer": answer_instance.strip()
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

    print(f"Dataset saved to {file_path}")


def download_circuitbreaker_dataset(train):
    if train:
        file = "https://raw.githubusercontent.com/GraySwanAI/circuit-breakers/refs/heads/main/data/circuit_breakers_train.json"
    else:
        file = "https://raw.githubusercontent.com/GraySwanAI/circuit-breakers/refs/heads/main/data/circuit_breakers_val.json"
    file_path = os.path.join(RAW_DATA_DIR, f'circuitbreaker_{"train" if train else "val"}.json')
    
    download_file(file, file_path)
    print(f"Circuitbreaker {file_path} downloaded")


def process_circuitbreaker_dataset(train):
    raw_file_path = os.path.join(RAW_DATA_DIR, f'circuitbreaker_{"train" if train else "val"}.json')
    processed_file_path = os.path.join(PROCESSED_DATA_DIR, f'circuitbreaker_{"train" if train else "val"}.json')

    # dataset keys: category, prompt, llama3_output
    with open(raw_file_path, 'r') as f:
        dataset = json.load(f)
    
    data_evolver = DataEvolver()

    evolved_dataset = []
    for data in dataset:
        question = data['prompt']
        question_category = None if 'category' not in data else data['category']
        refusal_answer = data['llama3_output']
        output = data['output']
        evolved_question, evolved_answer = data_evolver.evol_data(question, question_category)
        evolved_dataset.append({
            "category": question_category,
            "evolved_question": evolved_question,
            "evolved_answer": evolved_answer,
            "refusal_answer": refusal_answer,
            "output": output,
            "question": question
        })
    dump_json(evolved_dataset, processed_file_path)
    print("Evolved train dataset saved to ", processed_file_path)

#-------------------------Main-------------------------
def main():
    download_circuitbreaker_dataset(train=True)
    download_circuitbreaker_dataset(train=False)
    process_circuitbreaker_dataset(train=True) 
    process_circuitbreaker_dataset(train=False) 


if __name__ == "__main__":
    main()