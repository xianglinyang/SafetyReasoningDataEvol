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

import asyncio
import os
import json
import logging
from tqdm import tqdm
import argparse

from src.evol.question_evol import QuestionEvol
from src.llm_zoo import load_model
from src.logger.config import setup_logging

logger = logging.getLogger(__name__)

# -------------------------Data Loading and Saving-------------------------

PROCESSED_DATA_DIR = "data/processed"
RAW_DATA_DIR = "data/raw"

def dump_json(dataset, file_path):
    dir = os.path.dirname(file_path)
    if not os.path.exists(dir):
        os.makedirs(dir)

    with open(file_path, "w") as file:
        json.dump(dataset, file, indent=4)

    logger.info(f"Dataset saved to {file_path}")

#-------------------------Data Processing-------------------------
async def mutate_dataset(dataset_name, llm_name, num_variants=18, alpha=0.5, demo_selected_strategy="diverse"):
    logger.info(f"Processing {dataset_name} dataset...")
    
    processed_file_path = os.path.join(PROCESSED_DATA_DIR, f'{dataset_name}_{demo_selected_strategy}.json')
    if os.path.exists(processed_file_path):
        with open(processed_file_path, 'r') as f:
            dataset = json.load(f)
    else:
        raw_file_path = os.path.join(RAW_DATA_DIR, f'{dataset_name}.json')
        with open(raw_file_path, 'r') as f:
            dataset = json.load(f)

    llm_client = load_model(llm_name)

    new_dataset = []
    count = 0

    questions = [data['instruction'] for data in dataset]

    evolver = QuestionEvol()
    question_variants = await evolver.generate_prompt_variants_batch(questions, llm_client, num_variants, alpha, demo_selected_strategy)

    for data, question_variant in tqdm(zip(dataset, question_variants), desc="Evolving dataset"):
        data['mutations'] = question_variant['mutations']
        count += 1
        new_dataset.append(data)
    
    dump_json(new_dataset, processed_file_path)
    logger.info(f"Mutated {dataset_name} dataset with question mutations saved to {processed_file_path}") 

#-------------------------Main-------------------------
async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str)
    parser.add_argument("--llm_name", type=str)
    parser.add_argument('-n', "--num_variants", type=int)
    parser.add_argument("--alpha", type=float)
    parser.add_argument("--demo_selected_strategy", type=str, default="diverse")
    parser.add_argument("--run_id", '-r', type=int)
    
    args = parser.parse_args()
    dataset_name = args.dataset_name
    llm_name = args.llm_name
    num_variants = args.num_variants
    alpha = args.alpha
    demo_selected_strategy = args.demo_selected_strategy
    run_id = args.run_id

    setup_logging(task_name="data_evol", run_id=run_id)
    logger.info("Assembling data...")

    logger.info(f"Mutating {dataset_name} dataset...")
    await mutate_dataset(dataset_name, llm_name, num_variants, alpha, demo_selected_strategy)
    
if __name__ == "__main__":    
    asyncio.run(main())