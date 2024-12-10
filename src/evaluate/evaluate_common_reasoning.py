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

import torch
import time
import random
from tqdm import tqdm
import argparse

from src.llm_zoo.code_base_models import HuggingFaceLLM
from src.data_utils.reasoning_datasets import ReasoningDataset, gt_answer_cleansing, answer_cleansing, zero_shot_answer_trigger

def evaluate_reasoning(llm, dataset_name, dataset, eval_num=-1):
    correct = 0
    t0 = time.time()

    if eval_num == -1:
        eval_idxs = range(len(dataset))
    else:
        eval_idxs = random.sample(range(len(dataset)), eval_num)

    for idx in tqdm(eval_idxs):
        question, _, answer = dataset[idx]
        llm_answer = llm.invoke(question)

        print(f"{idx} Question:")
        print(question, "\n")
        pred_answer = answer_cleansing(dataset_name, llm_answer)
        answer = gt_answer_cleansing(dataset_name, answer)
        print(answer == pred_answer, f"[GT answer: {answer} Pred answer: {pred_answer}]\n\n")
        correct += (answer == pred_answer)

    t1 = time.time()
    print('*****************************')
    print(f'Evaluate num:{eval_num}')
    print(f"Accuracy: {correct/eval_num} = {correct}/{eval_num}")
    print(f"Time spent: {(t1-t0)/60:.2f} minutes")


def main():
    # parse args from command line
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--model_name_or_path", type=str, default="out/checkpoint-312")
    # parser.add_argument("--dataset_name", type=str, default="gsm8k")
    # parser.add_argument("--split", type=str, default="test")
    # parser.add_argument("--eval_num", type=int, default=10)
    # parser.add_argument("--device", type=str, default="cuda")
    # args = parser.parse_args()

    # model_name_or_path = args.model_path

    model_name_or_path = "out"
    model_abbr = "llama2"
    torch_type = torch.bfloat16
    dataset_name = "gsm8k"
    split = "test"
    eval_num = 100
    device = "cuda"
    llm = HuggingFaceLLM(model_name_or_path=model_name_or_path, model_abbr=model_abbr, torch_dtype=torch_type, device=device)
    dataset = ReasoningDataset(dataset_name=dataset_name, split=split)
    evaluate_reasoning(llm, dataset_name, dataset, eval_num)

    model_name_or_path = "meta-llama/Llama-2-7b-chat-hf"
    model_abbr = "llama2"
    torch_type = torch.bfloat16
    dataset_name = "gsm8k"
    split = "test"
    eval_num = 100
    device = "cuda"
    llm = HuggingFaceLLM(model_name_or_path=model_name_or_path, model_abbr=model_abbr, torch_dtype=torch_type, device=device)
    dataset = ReasoningDataset(dataset_name=dataset_name, split=split)
    evaluate_reasoning(llm, dataset_name, dataset, eval_num)

if __name__ == "__main__":
    main()
