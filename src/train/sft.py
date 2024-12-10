'''
The training script for the Safety Reasoning Data Evol model with LoRA.
'''

import os
import sys
import time
import logging
from typing import Optional

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    set_seed,
    Trainer,
    DataCollatorWithPadding
)
from peft import get_peft_model, LoraConfig, TaskType, PeftModel

from src.train.get_arguments import ModelArguments, DataArguments, TrainingArguments
from src.data_utils.safety_datasets import SafetyReasoningDataset

logger = logging.getLogger(__name__)

# redirect logging from console to file
def setup_logging(training_args):
    """Setup logging to both file and console"""
    
    # Create logs directory if it doesn't exist
    log_dir = os.path.join(training_args.output_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    
    # Create log file path with timestamp
    time_stamp = time.strftime("%Y%m%d-%H%M%S")
    log_file = os.path.join(log_dir, f"training_{time_stamp}.log")
    
    # Setup logging format
    log_format = "%(asctime)s - %(levelname)s - %(name)s - %(message)s"
    date_format = "%m/%d/%Y %H:%M:%S"
    
    # Setup logging to both file and console
    logging.basicConfig(
        format=log_format,
        datefmt=date_format,
        level=logging.INFO,
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    logger.setLevel(logging.INFO)

def compute_loss(model, inputs, return_outputs=False):
    # Print input shapes for debugging
    print("Input shapes:", {k: v.shape for k, v in inputs.items()})
    outputs = model(**inputs)
    loss = outputs.loss
    return (loss, outputs) if return_outputs else loss

def main():
    # Parse arguments
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    
    # Setup logging
    setup_logging(training_args)
    
    # Set seed for reproducibility
    set_seed(training_args.seed)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        use_fast=model_args.use_fast_tokenizer,
        padding_side="right"
    )
    logger.info(f"Training parameters {training_args}")
    logger.info(f"Model parameters {model_args}")
    logger.info(f"Dataset parameters {data_args}")

    # Set seed before initializing model.
    set_seed(training_args.seed)

    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "<pad>"})
    model = AutoModelForCausalLM.from_pretrained(model_args.model_name_or_path, torch_dtype=model_args.torch_dtype, device_map=model_args.device_map)

    # Load training dataset
    train_dataset = SafetyReasoningDataset(
        dataset_name=data_args.dataset_name,
        split="train",
        tokenizer=tokenizer,
        model_name=model_args.model_name_abbr,
        max_length=data_args.max_seq_length
        )
    val_dataset = SafetyReasoningDataset(
        dataset_name=data_args.dataset_name,
        split="val",
        tokenizer=tokenizer,
        model_name=model_args.model_name_abbr,
        max_length=data_args.max_seq_length
    )

    # resize embeddings if needed (e.g. for LlamaTokenizer)
    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))
        # if you load lora model and resize the token embeddings, the requires_grad flag is set to True for embeddings
        if isinstance(model, PeftModel):
            model.get_input_embeddings().weight.requires_grad = False
            model.get_output_embeddings().weight.requires_grad = False

    if not isinstance(model, PeftModel) and model_args.lora:
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=model_args.lora_r,
            lora_alpha=model_args.lora_alpha,
            lora_dropout=model_args.lora_dropout,
            target_modules=model_args.lora_target_modules,
        )
        model = get_peft_model(model, lora_config)
        logger.info(
            f"Applied LoRA to model."
        )
        model.print_trainable_parameters()

        # for checkpointing
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)


    model_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"trainable model_params: {model_params}")

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(
            tokenizer=tokenizer,
            padding=True,
            pad_to_multiple_of=8
        )
    )
    
    # Training
    logger.info("*** Starting training ***")
    train_result = trainer.train()
    
    # Save final model
    logger.info("*** Saving final model ***")
    trainer.save_model(training_args.output_dir)
    
    # Save training metrics
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    
    # Save trainer state
    trainer.save_state()
    
    # Reload and save tokenizer with added tokens
    tokenizer.save_pretrained(training_args.output_dir)

if __name__ == "__main__":
    # TODO: remove wandb
    #---wandb---
    import wandb
    wandb.init(mode="offline", dir="out")  # Saves logs locally in outputs/wandb
    # Or alternatively, you can set the environment variable
    os.environ["WANDB_MODE"] = "offline"
    os.environ["WANDB_DIR"] = "out"
    #---wandb---
    main()