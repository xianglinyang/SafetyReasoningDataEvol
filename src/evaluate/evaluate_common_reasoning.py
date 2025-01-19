'''
Evaluate the loss of the fine-tuned model on different general capability datasets


Open LLM Leaderboard
MT benchmark
MMLU (Hendrycks et al., 2020), 
ARC (Clark et al., 2018), 
GSM8K (Cobbe et al., 2021), and
TRUTHFULQA (Bisk et al., 2020)
BBH (Bisk et al., 2020)
humaneval
alpaca
'''

import os
import sys
import json
import time
from datetime import datetime
import random
from tqdm import tqdm
import logging
import torch
import argparse
import fcntl
import asyncio
from typing import Dict


from src.logger.config import setup_logging
from src.llm_zoo.code_base_models import HuggingFaceLLM
from src.data_utils.reasoning_datasets import ReasoningDataset, gt_answer_cleansing, answer_cleansing_with_llm

logger = logging.getLogger(__name__)

def process_single_evaluation(llm, dataset_name, question, gt_answer, idx):
    trigger = "Solve the following problem:\n\n"
    llm_answer = llm.invoke(trigger + question)
    
    generated_answers = llm_answer.split("#### Response")
    reasoning = generated_answers[0] if len(generated_answers) > 1 else ""
    generated_answer = generated_answers[-1]
    
    logger.info(f"{idx} Question: {question}")
    logger.info(f"[Reasoning]: {reasoning}")
    logger.info(f"[LLM Answer]: {generated_answer}")

    pred_answer = answer_cleansing_with_llm(dataset_name, question, generated_answer)
    clean_answer = gt_answer_cleansing(dataset_name, gt_answer)
    logger.info(f"[GT answer]: {clean_answer}")
    logger.info(f"[Pred answer]: {pred_answer}")
    logger.info(f"[Cleaning]: {clean_answer == pred_answer}")
    
    return int(clean_answer == pred_answer)

def evaluate_reasoning(llm, dataset_name, dataset, eval_num=-1):
    t0 = time.time()

    if eval_num == -1:
        eval_idxs = list(range(len(dataset)))
    elif eval_num > len(dataset):
        eval_idxs = list(range(len(dataset)))
        eval_num = len(dataset)
    else:
        eval_idxs = random.sample(range(len(dataset)), eval_num)

    correct = [0] * len(eval_idxs)
    
    # Process evaluations sequentially
    for i, idx in enumerate(tqdm(eval_idxs)):
        question, _, answer = dataset[idx]
        correct[i] = process_single_evaluation(llm, dataset_name, question, answer, idx)

    return sum(correct) / len(correct), time.time() - t0

def save_results(results: Dict, path="eval_results"):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    
    save_file = os.path.join(path, "common_ability.json")
    max_retries = 5
    retry_delay = 1
    
    for attempt in range(max_retries):
        try:
            # Open in r+ mode (read and write without truncating)
            with open(save_file, 'r+' if os.path.exists(save_file) else 'w+') as f:
                # Acquire lock before doing anything
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                try:
                    try:
                        # Read existing content
                        f.seek(0)  # Ensure we're at the start of file
                        existing_evaluation = json.load(f)
                    except (ValueError, json.JSONDecodeError):
                        # Handle empty or invalid file
                        existing_evaluation = []
                    
                    # Append new results
                    existing_evaluation.append(results.copy())
                    
                    # Write back entire content
                    f.seek(0)  # Go back to start
                    f.truncate()  # Clear existing content
                    json.dump(existing_evaluation, f, indent=4)
                    
                    print(f"Evaluation results saved at {save_file}")
                    return True
                    
                finally:
                    # Release the lock
                    print("Releasing lock...")
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)
                    print("Lock released")
                    
        except Exception as e:
            if attempt == max_retries - 1:
                print(f"Failed to save results after {max_retries} attempts: {e}")
                return False
            time.sleep(retry_delay)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str)
    parser.add_argument("--dataset_name", type=str, default="gsm8k")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--eval_num", type=int, default=-1)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--torch_type", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--run_id", type=str, default=None)
    args = parser.parse_args()

    setup_logging(task_name="evaluate_loss", run_id=args.run_id)

    # log the args
    logger.info(f"Arguments: {args}")

    # read args
    model_name_or_path = args.model_name_or_path
    torch_type = args.torch_type
    dataset_name = args.dataset_name
    split = args.split
    eval_num = args.eval_num
    device = args.device

    if torch_type == "bf16":
        torch_type = torch.bfloat16
    elif torch_type == "fp16":
        torch_type = torch.float16
    elif torch_type == "fp32":
        torch_type = torch.float32
    else:
        raise ValueError(f"Invalid torch_type: {torch_type}")

    llm = HuggingFaceLLM(model_name_or_path=model_name_or_path, torch_dtype=torch_type, device=device)
    dataset = ReasoningDataset(dataset_name=dataset_name, split=split)
    accu, elapsed_time = evaluate_reasoning(llm, dataset_name, dataset, eval_num)

    results = {
        "accu": accu,
        "dataset_name": dataset_name,
        "model_name_or_path": model_name_or_path,
        "split": split,
        "eval_num": eval_num,
        "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "elapsed_time": elapsed_time,
        "average_time_per_sample": elapsed_time / eval_num
    }
    logger.info(f"Evaluation results: {results}")
    save_results(results)
    print("End of evaluation")


if __name__ == "__main__":
    main()
