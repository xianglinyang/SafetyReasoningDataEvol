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
from src.llm_zoo.code_base_models import VLLMModel
from src.data_utils.reasoning_datasets import ReasoningDataset, batch_answer_cleansing_with_llm, batch_gt_answer_cleansing

logger = logging.getLogger(__name__)

async def process_evaluation(llm, dataset_name, questions, gt_answers, clean_model_name="gpt-4.1-nano"):
    # 1. get the generated answers
    trigger = "Solve the following problem:\n\n"
    processed_questions = [trigger + question for question in questions]

    print("Example questions:")
    print(processed_questions[:3])
    
    try:
        llm_answers, latency_metrics = llm.batch_invoke(processed_questions, return_latency=True)
    except Exception as e:
        logger.error(f"Error during LLM batch invocation: {e}")
        return [False] * len(questions)  # Return all False if LLM fails

    # 2. clean the generated answers
    reasoning_list = []
    generated_answer_list = []
    for llm_answer in llm_answers:
        split_llm_answer = llm_answer.split("#### Response")
        if len(split_llm_answer) > 1:
            reasoning = split_llm_answer[0].strip()
            generated_answer = split_llm_answer[1].strip()
        else:
            # If no "#### Response" found, treat the entire answer as the response
            reasoning = ""
            generated_answer = llm_answer.strip()
        reasoning_list.append(reasoning)
        generated_answer_list.append(generated_answer)
    
    print("Example reasoning:")
    print(reasoning_list[:3])
    print("Example generated answer:")
    print(generated_answer_list[:3])
    print("Example gt answer:")
    print(gt_answers[:3])

    # 3. clean the generated answers
    try:
        pred_answer_list = await batch_answer_cleansing_with_llm(dataset_name, questions, generated_answer_list, clean_model_name)
        clean_answer_list = batch_gt_answer_cleansing(dataset_name, gt_answers)
    except Exception as e:
        logger.error(f"Error during answer cleansing: {e}")
        return [False] * len(questions)  # Return all False if cleansing fails

    # 4. calculate the accuracy
    corrects = [clean_answer == pred_answer for clean_answer, pred_answer in zip(clean_answer_list, pred_answer_list)]
    return corrects, latency_metrics


async def evaluate_reasoning(llm, dataset_name, dataset, eval_num=-1, clean_model_name="gpt-4.1-nano"):
    if eval_num == -1:
        eval_idxs = list(range(len(dataset)))
    elif eval_num > len(dataset):
        eval_idxs = list(range(len(dataset)))
        eval_num = len(dataset)
    else:
        eval_idxs = random.sample(range(len(dataset)), eval_num)

    questions = [dataset[idx][0] for idx in eval_idxs]
    gt_answers = [dataset[idx][2] for idx in eval_idxs]

    corrects, latency_metrics = await process_evaluation(llm, dataset_name, questions, gt_answers, clean_model_name)

    # Fix division by zero issue
    if not corrects:
        return 0.0
    return sum(corrects) / len(corrects), latency_metrics


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


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str)
    parser.add_argument("--dataset_name", type=str, default="gsm8k")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--eval_num", type=int, default=-1)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--torch_type", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--run_id", type=str, default=None)
    parser.add_argument("--llm_model_name", type=str, default="gpt-4.1-mini")
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
    tensor_parallel_size = args.tensor_parallel_size
    clean_model_name = args.llm_model_name
    if torch_type == "bf16":
        torch_type = torch.bfloat16
    elif torch_type == "fp16":
        torch_type = torch.float16
    elif torch_type == "fp32":
        torch_type = torch.float32
    else:
        raise ValueError(f"Invalid torch_type: {torch_type}")

    llm = VLLMModel(model_name_or_path=model_name_or_path, torch_dtype=torch_type, device=device, tensor_parallel_size=tensor_parallel_size)
    dataset = ReasoningDataset(dataset_name=dataset_name, split=split)
    
    # Determine actual evaluation number
    if eval_num == -1:
        actual_eval_num = len(dataset)
    elif eval_num > len(dataset):
        actual_eval_num = len(dataset)
    else:
        actual_eval_num = eval_num
    
    accu, latency_metrics = await evaluate_reasoning(llm, dataset_name, dataset, eval_num, clean_model_name)

    results = {
        "accu": accu,
        "dataset_name": dataset_name,
        "model_name_or_path": model_name_or_path,
        "split": split,
        "eval_num": actual_eval_num,
        "tensor_parallel_size": tensor_parallel_size,
        "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "latency_metrics": latency_metrics,
    }
    logger.info(f"Evaluation results: {results}")
    save_results(results)
    print("End of evaluation")


if __name__ == "__main__":
    asyncio.run(main())
