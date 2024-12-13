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
import fcntl


from src.logger.config import setup_logging
from src.llm_zoo.code_base_models import HuggingFaceLLM
from src.data_utils.reasoning_datasets import ReasoningDataset, gt_answer_cleansing, answer_cleansing_with_llm, zero_shot_answer_trigger

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
        pred_answer = answer_cleansing_with_llm(dataset_name, llm_answer)
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


def save_results(results, save_path="eval_results"):
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)
    
    save_file = os.path.join(save_path, "common_ability.json")
    max_retries = 5
    retry_delay = 1
    
    for attempt in range(max_retries):
        try:
            with open(save_file, 'a+') as f:  # Use a+ mode to create if not exists
                # Acquire an exclusive lock
                fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                try:
                    try:
                        existing_evaluation = json.load(f)
                    except (json.JSONDecodeError, ValueError):
                        # File is empty or invalid JSON
                        existing_evaluation = list()
                    
                    # Add new results
                    existing_evaluation.append(results)
                    json.dump(existing_evaluation, f)

                    logger.info(f"Evaluation results saved at {save_file}")
                    return True
                finally:
                    # Release the lock
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)
        except Exception as e:
            if attempt == max_retries - 1:
                logger.error(f"Failed to save results after {max_retries} attempts: {e}")
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
    accu, correct, eval_idxs = evaluate_reasoning(llm, dataset_name, dataset, eval_num)

    results = {
        "accu": accu,
        "dataset_name": dataset_name,
        "model_name_or_path": model_name_or_path,
        "split": split,
        "eval_num": eval_num
    }
    logger.info(f"Evaluation results: {results}")
    save_results(results)


if __name__ == "__main__":
    main()
