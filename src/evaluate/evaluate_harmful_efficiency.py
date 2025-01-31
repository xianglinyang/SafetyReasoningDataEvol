'''
Evaluate the model's ability to refuse to answer harmful questions
1. None (only harmful questions)
2. prompt-based (prompt-based harmful questions)
3. adversarial (adversarial harmful questions)

Metrics:
1. ASR with substring matching
2. Safety score with llama-guard2
'''
from typing import List, Dict
import json
import os
import random
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import time
import litellm
import argparse
import logging
import fcntl
import asyncio
from datetime import datetime

from src.logger.config import setup_logging
from src.data_utils.harmful_datasets import HarmfulDataset
from src.llm_zoo.code_base_models import HuggingFaceLLM
from src.llm_zoo.api_base_models import OpenAILLM
from src.evaluate.prompt_attack import get_attack_fn, __prompt_attacks_methods__

logger = logging.getLogger(__name__)

def get_completions(llm, dataset, attack_name, eval_num=-1):
    logger.info(f"Getting completions for {attack_name} attack")
    attack_fn = get_attack_fn(attack_name)

    if eval_num == -1:
        eval_idxs = range(len(dataset))
    elif eval_num > len(dataset):
        eval_idxs = range(len(dataset))
        eval_num = len(dataset)
    else:
        eval_idxs = random.sample(range(len(dataset)), eval_num)

    t0 = time.time()
    for idx in tqdm(eval_idxs):
        question, _ = dataset[idx]
        attack_question = attack_fn(question)
        llm_answer = llm.invoke(attack_question)
    t1 = time.time()

    total_time = t1 - t0
    average_time_per_sample = total_time / eval_num

    logger.info(f"Total time: {total_time}, Average time per sample: {average_time_per_sample}")

    return total_time, average_time_per_sample

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str)
    parser.add_argument("--torch_type", type=str, choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--dataset_name", type=str, default="sorrybench")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--eval_num", type=int, default=-1)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--attack_name", type=str)
    parser.add_argument("--run_id", type=str, help="Unique identifier for this run for logging")
    args = parser.parse_args()

    setup_logging(task_name="evaluate_harmful", run_id=args.run_id)

    logger.info(f"Evaluating Arguments: {args}")
    
    model_name_or_path = args.model_name_or_path
    dataset_name = args.dataset_name
    split = args.split
    eval_num = args.eval_num
    device = args.device
    attack_name = args.attack_name

    torch_type = args.torch_type
    if torch_type == "bf16":
        torch_type = torch.bfloat16
    elif torch_type == "fp16":
        torch_type = torch.float16
    elif torch_type == "fp32":
        torch_type = torch.float32
    else:
        raise ValueError(f"Invalid torch_type: {torch_type}")

    llm = HuggingFaceLLM(model_name_or_path=model_name_or_path, torch_dtype=torch_type, device=device)
    dataset = HarmfulDataset(dataset_name=dataset_name, split=split)
    total_time, average_time_per_sample = get_completions(llm, dataset, attack_name, eval_num)

    logger.info(f"Total time: {total_time}, Average time per sample: {average_time_per_sample}")
    logger.info("End of evaluation")


if __name__ == "__main__":
    main()



