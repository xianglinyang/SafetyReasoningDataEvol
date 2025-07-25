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

def process_evaluation(llm, dataset_name, questions, gt_answers, clean_model_name="gpt-4.1-nano"):
    # 1. get the generated answers
    trigger = "Solve the following problem:\n\n"
    processed_questions = [trigger + question for question in questions]
    llm_answers = llm.batch_invoke(processed_questions)

    # 2. clean the generated answers
    reasoning_list = []
    generated_answer_list = []
    for llm_answer in llm_answers:
        split_llm_answer = llm_answer.split("#### Response")
        reasoning = split_llm_answer[0] if len(split_llm_answer) > 1 else ""
        generated_answer = split_llm_answer[-1]
        reasoning_list.append(reasoning)
        generated_answer_list.append(generated_answer)

    # 3. clean the generated answers
    pred_answer_list = batch_answer_cleansing_with_llm(dataset_name, questions, generated_answer_list, clean_model_name)
    clean_answer_list = batch_gt_answer_cleansing(dataset_name, gt_answers)

    # 4. calculate the accuracy
    corrects = [clean_answer == pred_answer for clean_answer, pred_answer in zip(clean_answer_list, pred_answer_list)]
    return corrects


def evaluate_reasoning(llm, dataset_name, dataset, eval_num=-1, clean_model_name="gpt-4.1-nano"):
    t0 = time.time()

    if eval_num == -1:
        eval_idxs = list(range(len(dataset)))
    elif eval_num > len(dataset):
        eval_idxs = list(range(len(dataset)))
        eval_num = len(dataset)
    else:
        eval_idxs = random.sample(range(len(dataset)), eval_num)

    questions = [dataset[idx][0] for idx in eval_idxs]
    gt_answers = [dataset[idx][2] for idx in eval_idxs]

    corrects = process_evaluation(llm, dataset_name, questions, gt_answers, clean_model_name)

    return sum(corrects) / len(corrects), time.time() - t0


def evaluate_reasoning_efficiency(llm, dataset, eval_num=-1):
    if eval_num == -1:
        eval_idxs = list(range(len(dataset)))
    elif eval_num > len(dataset):
        eval_idxs = list(range(len(dataset)))
        eval_num = len(dataset)
    else:
        eval_idxs = random.sample(range(len(dataset)), eval_num)
    
    # Process evaluations sequentially
    t0 = time.time()
    for i, idx in enumerate(tqdm(eval_idxs)):
        question, _, answer = dataset[idx]
        trigger = "Solve the following problem:\n\n"
        llm.invoke(trigger + question)
    t1 = time.time()

    total_time = t1 - t0
    average_time_per_sample = total_time / eval_num

    return total_time, average_time_per_sample

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
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--torch_type", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--run_id", type=str, default=None)
    parser.add_argument("--llm_model_name", type=str, default="gpt-4.1-nano")
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
    accu, elapsed_time = evaluate_reasoning(llm, dataset_name, dataset, eval_num, clean_model_name)

    results = {
        "accu": accu,
        "dataset_name": dataset_name,
        "model_name_or_path": model_name_or_path,
        "split": split,
        "eval_num": eval_num,
        "tensor_parallel_size": tensor_parallel_size,
        "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "elapsed_time": elapsed_time,
        "average_time_per_sample": elapsed_time / eval_num
    }
    logger.info(f"Evaluation results: {results}")
    save_results(results)
    print("End of evaluation")

def evaluate_reasoning_efficiency_main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str)
    parser.add_argument("--dataset_name", type=str, default="gsm8k")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--eval_num", type=int, default=-1)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--torch_type", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--run_id", type=str, default=None)
    args = parser.parse_args()

    setup_logging(task_name="evaluate_loss_efficiency", run_id=args.run_id)

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
    total_time, average_time_per_sample = evaluate_reasoning_efficiency(llm, dataset, eval_num)

    logger.info("Dataset name: %s, Model name: %s, Split: %s, Eval num: %d", dataset_name, model_name_or_path, split, eval_num)
    logger.info(f"Total time: {total_time}, Average time per sample: {average_time_per_sample}")


if __name__ == "__main__":
    main()
