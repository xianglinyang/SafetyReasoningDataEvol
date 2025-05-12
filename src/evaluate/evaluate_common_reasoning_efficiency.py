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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str)
    parser.add_argument("--dataset_name", type=str, default="gsm8k")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--eval_num", type=int, default=-1)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--torch_type", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--output_dir", type=str, default="results")
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
    output_dir = args.output_dir

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

    logger.info("End of evaluation")

    logger.info("Dataset name: %s, Model name: %s, Split: %s, Eval num: %d", dataset_name, model_name_or_path, split, eval_num)
    logger.info(f"Total time: {total_time}, Average time per sample: {average_time_per_sample}")

    # Save metrics to file
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(output_dir, f"efficiency_metrics_{model_name_or_path.replace('/', '-')}_{dataset_name}_{split}_{timestamp}.json")

    metrics = {
        "timestamp": timestamp,
        "dataset_name": dataset_name,
        "model_name_or_path": model_name_or_path,
        "split": split,
        "eval_num": eval_num,
        "total_time": total_time,
        "average_time_per_sample": average_time_per_sample
    }
    
    with open(output_file, 'w') as f:
        json.dump(metrics, f, indent=2)
  

if __name__ == "__main__":
    main()
