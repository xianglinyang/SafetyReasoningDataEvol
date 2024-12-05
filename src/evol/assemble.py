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
from src.evol.answers import answer_format

#-------------------------Data evolution-------------------------
def evol_question(question):
    question_strategy = QuestionStrategy()

    strategy = question_strategy.sample_strategy()
    question_instance = question_strategy.enrich_question_with_strategy(question, strategy)
    print("Question instance: ", question_instance)

    return question_instance


def evol_answer(question, question_category):
    answer_instance = answer_format.format(question=question, category=question_category)
    print("Answer instance: ", answer_instance)
    return answer_instance


def evol_data(question, question_category):
    question_instance = evol_question(question)
    answer_instance = evol_answer(question, question_category)
    return question_instance, answer_instance


def evol_dataset(dataset):
    evolved_dataset = []
    for question, question_category in dataset:
        question_instance, answer_instance = evol_data(question, question_category)
        evolved_dataset.append({"question": question_instance.strip(), "category": question_category, "answer": answer_instance})
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
    
    evolved_dataset = []
    for data in dataset:
        question = data['prompt']
        question_category = data['category']
        refusal_answer = data['llama3_output']
        output = data['output']
        evolved_question, evolved_answer = evol_data(question, question_category)
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
    # test function
    question = "What is the capital of France?"
    question_category = "commen knowledge"
    question_instance = evol_question(question)
    answer_instance = evol_answer(question, question_category)


if __name__ == "__main__":
    main()