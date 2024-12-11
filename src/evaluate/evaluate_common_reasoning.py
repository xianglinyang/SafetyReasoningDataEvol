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
import random
from tqdm import tqdm
import logging
import torch
import argparse

from src.llm_zoo.code_base_models import HuggingFaceLLM
from src.data_utils.reasoning_datasets import ReasoningDataset, gt_answer_cleansing, answer_cleansing, zero_shot_answer_trigger

logger = logging.getLogger(__name__)

def evaluate_reasoning(llm, dataset_name, dataset, eval_num=-1):
    t0 = time.time()

    if eval_num == -1:
        eval_idxs = list(range(len(dataset)))
    elif eval_num > len(dataset):
        eval_idxs = list(range(len(dataset)))
        eval_num = len(dataset)
    else:
        eval_idxs = random.sample(range(len(dataset)), eval_num)

    correct = [0]*len(eval_idxs)
    trigger = zero_shot_answer_trigger(dataset_name)

    for i in tqdm(range(len(eval_idxs))):
        idx = eval_idxs[i]
        question, _, answer = dataset[idx]
        llm_answer = llm.invoke(question+" "+trigger)

        logger.info(f"{idx} Question: {question}")
        logger.info(f"[GT answer]: {answer}")
        logger.info(f"[LLM Answer]: {llm_answer}")
        pred_answer = answer_cleansing(dataset_name, llm_answer)
        answer = gt_answer_cleansing(dataset_name, answer)
        logger.info(f"[Pred answer]: {pred_answer}")
        logger.info(f"[Cleaning]: {answer == pred_answer}")
        correct[i] = int(answer == pred_answer)

    t1 = time.time()

    time_spent = (t1-t0)/60
    accu = sum(correct)/eval_num
    print("*"*50)
    logger.info(f'Evaluate num:{eval_num}')
    logger.info(f"Accuracy: {accu} = {sum(correct)}/{eval_num}")
    logger.info(f"Time spent: {time_spent:.2f} minutes")
    return accu, correct, eval_idxs


def save_results(accu, dataset_name, model_name_or_path, split, eval_num, save_path="eval_results"):
    common_ability_path = f"{save_path}/common_ability.json"
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)

    if os.path.exists(common_ability_path):
        with open(common_ability_path, "r") as f:
            common_ability = json.load(f)
    else:
        common_ability = list()

    results = {
        "accu": accu,
        "dataset_name": dataset_name,
        "model_name_or_path": model_name_or_path,
        "split": split,
        "eval_num": eval_num
    }
    common_ability.append(results)
    
    with open(common_ability_path, "w") as f:
        json.dump(common_ability, f)
    return results
    

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str)
    parser.add_argument("--dataset_name", type=str, default="gsm8k")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--eval_num", type=int, default=-1)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--torch_type", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])
    args = parser.parse_args()

    # log the args
    logger.info(f"Arguments: {args}")

    # read args
    model_name_or_path = args.model_path
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
    accu, correct, eval_idxs = evaluate_reasoning(llm, dataset_name, dataset, eval_num)
    save_results(accu, dataset_name, model_name_or_path, split, eval_num)

def test():
    # test input
    model_name_or_path = "out"
    torch_type = torch.bfloat16
    dataset_name = "mmlu"
    split = "test"
    eval_num = 5
    device = "cuda:0"

    llm = HuggingFaceLLM(model_name_or_path=model_name_or_path, torch_dtype=torch_type, device=device)
    dataset = ReasoningDataset(dataset_name=dataset_name, split=split)
    accu, correct, eval_idxs = evaluate_reasoning(llm, dataset_name, dataset, eval_num)
    save_results(accu, dataset_name, model_name_or_path, split, eval_num)

if __name__ == "__main__":
    from src.logger.config import setup_logging
    setup_logging(task_name="evaluate_loss")
    main()
