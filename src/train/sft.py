'''
The training script for the Safety Reasoning Data Evol model with LoRA.
'''

import os
import logging
import torch.distributed as dist
import torch

from transformers import (
    HfArgumentParser,
    set_seed,
    Trainer,
    DataCollatorForSeq2Seq
)

from src.train.get_arguments import ModelArguments, DataArguments, TrainingArguments
from src.data_utils.safety_datasets import SafetyReasoningDataset
from src.logger.config import setup_logging
from src.logger.train_log import LoggingCallback
from src.utils.train_utils import load_tokenizer_and_model, save_tokenizer_and_model
from src.llm_zoo.model_configs import get_system_prompt


logger = logging.getLogger(__name__)

def main():
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

    # Load datasets for training and validation
    train_dataset = SafetyReasoningDataset(
        model_name=model_args.model_name_or_path,
        dataset_name=data_args.dataset_name,
        split="train",
        tokenizer=tokenizer,
        ratio=data_args.ratio,
        max_length=data_args.max_seq_length,
        # system_inst=get_system_prompt(model_args.model_name_or_path)
    )
    val_dataset = SafetyReasoningDataset(
        model_name=model_args.model_name_or_path,
        dataset_name=data_args.dataset_name,
        split="val",
        tokenizer=tokenizer,
        max_length=data_args.max_seq_length,
        ratio=data_args.ratio,
        # system_inst=get_system_prompt(model_args.model_name_or_path)
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=DataCollatorForSeq2Seq(
            tokenizer=tokenizer, model=model, padding="longest"),
        callbacks=[LoggingCallback()]
    )
    
    # Training
    logger.info("*** Starting training ***")
    train_result = trainer.train()

    # save to output_dir
    save_tokenizer_and_model(model, tokenizer, training_args)

    # Cleanup distributed training
    if dist.is_initialized():
        dist.destroy_process_group()

if __name__ == "__main__":
    # TODO: remove wandb
    #---wandb---
    import os
    # Or alternatively, you can set the environment variable
    os.environ["WANDB_MODE"] = "offline"
    os.environ["WANDB_DIR"] = "."
    #---wandb---
    main()
