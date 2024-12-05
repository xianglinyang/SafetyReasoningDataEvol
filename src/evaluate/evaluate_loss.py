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
import logging
from dataclasses import dataclass, field
from typing import Optional, Dict

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    set_seed,
)
from datasets import load_dataset
from tqdm import tqdm

from src.train.get_arguments import ModelArguments, DataArguments

logger = logging.getLogger(__name__)

@dataclass
class EvalArguments:
    """Arguments for evaluation"""
    eval_datasets: str = field(
        default="mmlu,arc,gsm8k,truthfulqa,bbh",
        metadata={"help": "Comma-separated list of datasets to evaluate on"}
    )
    batch_size: int = field(
        default=1,
        metadata={"help": "Batch size for evaluation"}
    )
    max_samples: Optional[int] = field(
        default=None,
        metadata={"help": "Maximum number of samples to evaluate. None means all."}
    )
    output_dir: str = field(
        default="evaluation_results",
        metadata={"help": "Directory to save evaluation results"}
    )

def load_eval_dataset(dataset_name: str, max_samples: Optional[int] = None):
    """Load and preprocess evaluation dataset"""
    if dataset_name == "mmlu":
        dataset = load_dataset("cais/mmlu", "all")["test"]
    elif dataset_name == "arc":
        dataset = load_dataset("ai2_arc", "ARC-Challenge")["test"]
    elif dataset_name == "gsm8k":
        dataset = load_dataset("gsm8k", "main")["test"]
    elif dataset_name == "truthfulqa":
        dataset = load_dataset("truthful_qa", "multiple_choice")["validation"]
    elif dataset_name == "bbh":
        dataset = load_dataset("lukaemon/bbh")["test"]
    elif dataset_name == "humaneval":
        dataset = load_dataset("openai_humaneval")["test"]
    elif dataset_name == "alpaca":
        dataset = load_dataset("tatsu-lab/alpaca")["test"]
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    if max_samples is not None:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
    
    return dataset

def format_example(example: Dict, dataset_name: str) -> Dict:
    """Format example based on dataset type"""
    if dataset_name == "mmlu":
        question = f"Question: {example['question']}\nChoices:\nA) {example['choices'][0]}\nB) {example['choices'][1]}\nC) {example['choices'][2]}\nD) {example['choices'][3]}"
        answer = f"Answer: {example['answer']}"
    elif dataset_name == "arc":
        question = f"Question: {example['question']}\nChoices:\n" + "\n".join([f"{chr(65+i)}) {choice}" for i, choice in enumerate(example['choices'])])
        answer = f"Answer: {example['answerKey']}"
    elif dataset_name == "gsm8k":
        question = f"Question: {example['question']}"
        answer = f"Answer: {example['answer']}"
    elif dataset_name == "truthfulqa":
        question = f"Question: {example['question']}"
        answer = f"Answer: {example['correct_answers'][0]}"
    elif dataset_name == "bbh":
        question = f"Question: {example['input']}"
        answer = f"Answer: {example['target']}"
    elif dataset_name in ["humaneval", "alpaca"]:
        question = example["instruction"] if "instruction" in example else example["prompt"]
        answer = example["output"] if "output" in example else example["completion"]
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    return {"question": question, "answer": answer}

def format_and_tokenize(
    example: Dict,
    dataset_name: str,
    tokenizer: AutoTokenizer,
    max_length: int = 2048
) -> Dict:
    """Format example and tokenize according to model's chat template"""
    formatted = format_example(example, dataset_name)
    
    # Format as chat using model's chat template
    if hasattr(tokenizer, "apply_chat_template"):
        messages = [
            {"role": "user", "content": formatted["question"]},
            {"role": "assistant", "content": formatted["answer"]}
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False)
    else:
        # Fallback for models without chat template
        text = f"{formatted['question']}\n{formatted['answer']}"
    
    # Tokenize
    tokenized = tokenizer(
        text,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )
    
    # Create labels (same as input_ids, but -100 for non-assistant tokens)
    labels = tokenized.input_ids.clone()
    
    # If we have a chat template, find the assistant's response start
    if hasattr(tokenizer, "apply_chat_template"):
        assistant_tokens = tokenizer(
            tokenizer.apply_chat_template([{"role": "assistant", "content": ""}], tokenize=False),
            add_special_tokens=False
        ).input_ids
        
        # Find where assistant response starts
        response_start = None
        input_ids = tokenized.input_ids[0].tolist()
        for i in range(len(input_ids) - len(assistant_tokens)):
            if input_ids[i:i+len(assistant_tokens)] == assistant_tokens:
                response_start = i + len(assistant_tokens)
                break
        
        if response_start is not None:
            # Mask everything before assistant's response
            labels[0, :response_start] = -100
    
    return {
        "input_ids": tokenized.input_ids,
        "attention_mask": tokenized.attention_mask,
        "labels": labels
    }

def evaluate_dataset(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    dataset,
    dataset_name: str,
    batch_size: int = 1,
    device: str = "cuda",
    max_length: int = 2048
) -> Dict:
    """Evaluate model on a dataset"""
    model.eval()
    total_loss = 0
    total_samples = 0
    
    progress_bar = tqdm(range(0, len(dataset), batch_size), desc=f"Evaluating {dataset_name}")
    
    with torch.no_grad():
        for i in progress_bar:
            batch = dataset[i:i + batch_size]
            batch_losses = []
            
            for example in batch:
                encoded = format_and_tokenize(
                    example, 
                    dataset_name, 
                    tokenizer,
                    max_length
                )
                
                input_ids = encoded["input_ids"].to(device)
                attention_mask = encoded["attention_mask"].to(device)
                labels = encoded["labels"].to(device)
                
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                loss = outputs.loss
                batch_losses.append(loss.item())
            
            avg_batch_loss = sum(batch_losses) / len(batch_losses)
            total_loss += avg_batch_loss * len(batch)
            total_samples += len(batch)
            
            progress_bar.set_postfix({"loss": avg_batch_loss})
    
    return {
        "dataset": dataset_name,
        "average_loss": total_loss / total_samples,
        "num_samples": total_samples
    }

def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, EvalArguments))
    model_args, data_args, eval_args = parser.parse_args_into_dataclasses()
    
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    
    # Set seed
    set_seed(42)
    
    # Load model and tokenizer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        torch_dtype=model_args.torch_dtype,
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
    
    # Create output directory
    os.makedirs(eval_args.output_dir, exist_ok=True)
    
    # Evaluate on each dataset
    results = []
    for dataset_name in eval_args.eval_datasets.split(","):
        logger.info(f"Evaluating on {dataset_name}")
        
        try:
            # Load dataset
            dataset = load_eval_dataset(dataset_name, eval_args.max_samples)
            
            # Evaluate
            result = evaluate_dataset(
                model=model,
                tokenizer=tokenizer,
                dataset=dataset,
                dataset_name=dataset_name,
                batch_size=eval_args.batch_size,
                device=device
            )
            
            results.append(result)
            logger.info(f"Results for {dataset_name}: {result}")
            
        except Exception as e:
            logger.error(f"Error evaluating {dataset_name}: {str(e)}")
            continue
    
    # Save results
    output_file = os.path.join(eval_args.output_dir, "evaluation_results.json")
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Results saved to {output_file}")

if __name__ == "__main__":
    main()
