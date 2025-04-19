'''
Evaluate the model's ability to refuse to answer harmful questions
1. None (only harmful questions)
2. prompt-based (prompt-based harmful questions)
3. adversarial (adversarial harmful questions)

Metrics:
1. ASR with substring matching
2. Safety score with llama-guard2
3. Efficiency metrics: time, total tokens, reasoning tokens, answer tokens
'''
from typing import Tuple
import json
import os
import random
from tqdm import tqdm
import torch
import time
import argparse
import logging
from datetime import datetime

from src.logger.config import setup_logging
from src.data_utils.harmful_datasets import HarmfulDataset
from src.llm_zoo.code_base_models import HuggingFaceLLM
from src.llm_zoo.api_base_models import OpenAILLM
from src.evaluate.prompt_attack import get_attack_fn, __prompt_attacks_methods__

logger = logging.getLogger(__name__)

def split_reasoning_and_answer(response: str, delimiter: str = "#### Response:\n") -> Tuple[str, str]:
    """
    Split a response into reasoning and answer components based on a delimiter.
    
    Args:
        response: The model's response text
        delimiter: Text that signals the transition from reasoning to final answer
            Common options: "Therefore, ", "In conclusion, ", "My answer is: "
    
    Returns:
        Tuple of (reasoning_text, answer_text)
    """
    if delimiter not in response:
        return "", response
    reasoning, answer = response.split(delimiter, 1)
    reasoning = reasoning.strip()
    answer = answer.strip()
    return reasoning, answer


def count_tokens(tokenizer, text: str) -> int:
    """Count the number of tokens in the text using the specified tokenizer."""
    if not text:
        return 0
    return len(tokenizer.encode(text))

def get_completions(llm, dataset, attack_name, eval_num=-1, reasoning_delimiter="#### Response:\n"):
    logger.info(f"Getting completions for {attack_name} attack")
    attack_fn = get_attack_fn(attack_name)

    if eval_num == -1:
        eval_idxs = range(len(dataset))
    elif eval_num > len(dataset):
        eval_idxs = range(len(dataset))
        eval_num = len(dataset)
    else:
        eval_idxs = random.sample(range(len(dataset)), eval_num)

    # Metrics tracking
    t0 = time.time()
    total_input_tokens = 0
    total_output_tokens = 0
    total_reasoning_tokens = 0
    total_answer_tokens = 0
    
    metrics = {
        "per_sample": [],
        "overall": {}
    }

    for idx in tqdm(eval_idxs):
        sample_metrics = {}
        question, _ = dataset[idx]
        attack_question = attack_fn(question)
        
        # Count input tokens
        input_tokens = count_tokens(llm.tokenizer, attack_question)
        total_input_tokens += input_tokens
        sample_metrics["input_tokens"] = input_tokens
        
        # Time the inference
        sample_start = time.time()
        llm_answer = llm.invoke(attack_question)
        sample_time = time.time() - sample_start
        
        # Split reasoning and answer components
        reasoning_text, answer_text = split_reasoning_and_answer(llm_answer, reasoning_delimiter)
        
        # Count output tokens
        output_tokens = 0
        reasoning_tokens = 0
        answer_tokens = 0
        
        output_tokens = count_tokens(llm.tokenizer, llm_answer)
        total_output_tokens += output_tokens
        sample_metrics["output_tokens"] = output_tokens
            
        # Count tokens in reasoning and answer
        if reasoning_text:
            reasoning_tokens = count_tokens(llm.tokenizer, reasoning_text)
            total_reasoning_tokens += reasoning_tokens
            sample_metrics["reasoning_tokens"] = reasoning_tokens
        
        if answer_text:
            answer_tokens = count_tokens(llm.tokenizer, answer_text)
            total_answer_tokens += answer_tokens
            sample_metrics["answer_tokens"] = answer_tokens
        
        # Record metrics for this sample
        sample_metrics.update({
            "time_seconds": sample_time,
            "reasoning_text": reasoning_text,
            "answer_text": answer_text,
        })
        
        # Calculate ratios if both parts exist
        if reasoning_tokens > 0 and answer_tokens > 0:
            sample_metrics["reasoning_to_answer_ratio"] = reasoning_tokens / answer_tokens
        
        metrics["per_sample"].append(sample_metrics)
    
    t1 = time.time()
    total_time = t1 - t0
    average_time_per_sample = total_time / eval_num
    
    # Overall metrics
    metrics["overall"] = {
        "total_time_seconds": total_time,
        "average_time_per_sample": average_time_per_sample,
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "average_input_tokens": total_input_tokens / eval_num if total_input_tokens > 0 else 0,
        "average_output_tokens": total_output_tokens / eval_num if total_output_tokens > 0 else 0,
        "total_reasoning_tokens": total_reasoning_tokens,
        "total_answer_tokens": total_answer_tokens,
        "average_reasoning_tokens": total_reasoning_tokens / eval_num if total_reasoning_tokens > 0 else 0,
        "average_answer_tokens": total_answer_tokens / eval_num if total_answer_tokens > 0 else 0,
    }
    
    # Add ratio of reasoning to answer tokens
    if total_answer_tokens > 0:
        metrics["overall"]["reasoning_to_answer_ratio"] = total_reasoning_tokens / total_answer_tokens
    
    # Log results
    logger.info(f"Total time: {total_time}, Average time per sample: {average_time_per_sample}")
    if total_input_tokens > 0:
        logger.info(f"Total input tokens: {total_input_tokens}, Average per sample: {total_input_tokens/eval_num:.2f}")
    if total_output_tokens > 0:
        logger.info(f"Total output tokens: {total_output_tokens}, Average per sample: {total_output_tokens/eval_num:.2f}")
    if total_reasoning_tokens > 0:
        logger.info(f"Total reasoning tokens: {total_reasoning_tokens}, Average per sample: {total_reasoning_tokens/eval_num:.2f}")
    if total_answer_tokens > 0:
        logger.info(f"Total answer tokens: {total_answer_tokens}, Average per sample: {total_answer_tokens/eval_num:.2f}")
    if total_reasoning_tokens > 0 and total_answer_tokens > 0:
        logger.info(f"Reasoning to answer ratio: {total_reasoning_tokens/total_answer_tokens:.2f}")

    return metrics

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str)
    parser.add_argument("--torch_type", type=str, choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--dataset_name", type=str, default="sorrybench")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--eval_num", type=int, default=-1)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--attack_name", type=str)
    parser.add_argument("--output_dir", type=str, default="results")
    parser.add_argument("--reasoning_delimiter", type=str, default="#### Response:\n", 
                       help="Text that separates reasoning from final answer")
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
    output_dir = args.output_dir
    reasoning_delimiter = args.reasoning_delimiter

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
    metrics = get_completions(llm, dataset, attack_name, eval_num, reasoning_delimiter=reasoning_delimiter)

    # Save metrics to file
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(output_dir, f"efficiency_metrics_{model_name_or_path.replace('/', '-')}_{attack_name}_{timestamp}.json")
    
    with open(output_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    logger.info(f"Metrics saved to {output_file}")
    logger.info("End of evaluation")


if __name__ == "__main__":
    main()


# python src/evaluate/evaluate_harmful_efficiency.py \
#     --model_name_or_path your_model \
#     --attack_name none \
#     --output_dir results



