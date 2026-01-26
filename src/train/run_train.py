'''
The training script for the Safety Reasoning Data Evol model with LoRA.
'''

import os
import json
from datetime import datetime
import logging
import torch.distributed as dist
import torch
import random
import asyncio
import numpy as np

from transformers import (
    HfArgumentParser,
    set_seed
)

from src.train.get_arguments import ModelArguments, DataArguments, TrainingArguments
from src.train.cot_trainer import RobustCoTTrainer
from src.data_utils.RobustSCoT_datasets import SafetyReasoningDataset, SafetyDataCollator
from src.logger.config import setup_logging
from src.logger.train_log import LoggingCallback
from src.utils.train_utils import load_tokenizer_and_model, merge_lora_checkpoint
from src.llm_zoo import load_model

logger = logging.getLogger(__name__)

async def main():
    # Parse arguments
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Setup logging
    setup_logging(task_name="train", run_id=training_args.run_id)

    # log arguments
    logger.info(f"Training parameters {training_args}")
    logger.info(f"Model parameters {model_args}")
    logger.info(f"Dataset parameters {data_args}")
    
    # Set seed for reproducibility
    set_seed(training_args.seed)
    # for distributed training
    if model_args.device_map == "cuda" and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    
    tokenizer, model = load_tokenizer_and_model(model_args)
    
    # Store original values for outer loop
    num_outer_epochs = int(training_args.num_train_epochs)
    original_output_dir = training_args.output_dir


    # -------------------------------- datasets --------------------------------
    # load benign and harmful datasets
    from src.train.probe_and_select import load_processed_dataset
    benign_dataset, harmful_dataset = load_processed_dataset(data_args.probe_results_path)

    # shuffle the dataset
    random.shuffle(harmful_dataset)
    random.shuffle(benign_dataset)
    train_benign_dataset = benign_dataset[:int(len(benign_dataset)*0.995)]
    train_harmful_dataset = harmful_dataset[:int(len(harmful_dataset)*0.995)]
    val_benign_dataset = benign_dataset[int(len(benign_dataset)*0.995):]
    val_harmful_dataset = harmful_dataset[int(len(harmful_dataset)*0.995):]

    train_dataset = train_benign_dataset + train_harmful_dataset
    val_dataset = val_benign_dataset + val_harmful_dataset

    train_dataset = SafetyReasoningDataset(
        dataset=train_dataset,
        tokenizer=tokenizer,
        max_length=data_args.max_seq_length
    )
    val_dataset = SafetyReasoningDataset(
        dataset=val_dataset,
        tokenizer=tokenizer,
        max_length=data_args.max_seq_length
    )
        
    # -------------------------------- trainer --------------------------------
    # Calculate max_steps for 1 epoch only (not num_train_epochs)
    max_steps = len(train_dataset) // (training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps)
    
    trainer = RobustCoTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=SafetyDataCollator(tokenizer=tokenizer),
        callbacks=[LoggingCallback()],
        total_steps=max_steps,
        benign_lambda=model_args.benign_lambda,
        harmful_lambda=model_args.harmful_lambda,
    )
        
    # Training
    train_result = trainer.train()

    # -------------------------------- save checkpoint --------------------------------
    # Save checkpoint for this epoch to avoid overwriting
    # Create checkpoint directory in the format expected by merge_lora_checkpoint
    checkpoint_dir = os.path.join(original_output_dir, f"checkpoint-epoch-{training_args.probe_epoch}")
    adapter_dir = os.path.join(checkpoint_dir, "adapter_model")
    tokenizer_dir = os.path.join(checkpoint_dir, "tokenizer")
    
    os.makedirs(adapter_dir, exist_ok=True)
    os.makedirs(tokenizer_dir, exist_ok=True)
    
    logger.info(f"*** Saving checkpoint for epoch {training_args.probe_epoch + 1} to {checkpoint_dir} ***")
    
    # Save the model (LoRA adapter) and tokenizer in the expected structure
    model.save_pretrained(adapter_dir, safe_serialization=True)
    tokenizer.save_pretrained(tokenizer_dir, safe_serialization=True)

    logger.info(f"Checkpoint saved to {checkpoint_dir}")

    # -------------------------------- final model merging --------------------------------
    # After all epochs, merge all LoRA checkpoints if using LoRA
    if model_args.lora and training_args.merge_lora_after_training:
        logger.info("*** Merging all LoRA checkpoints ***")
        merge_lora_checkpoint(model_args.model_name_or_path, original_output_dir)
        logger.info("All LoRA checkpoints merged successfully")


if __name__ == "__main__":
    # TODO: remove wandb
    #---wandb---
    import os
    # Or alternatively, you can set the environment variable
    os.environ["WANDB_MODE"] = "offline"
    os.environ["WANDB_DIR"] = "."
    #---wandb---
    asyncio.run(main())

