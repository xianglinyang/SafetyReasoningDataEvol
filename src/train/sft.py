'''
The training script for the Safety Reasoning Data Evol model with LoRA.
'''

import os
import sys
import time
import logging
import torch.distributed as dist
from transformers.trainer_utils import get_last_checkpoint
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
from src.logger.config import setup_logging
from src.utils.dtype_utils import str2dtype

logger = logging.getLogger(__name__)

def main():
    # Setup logging
    setup_logging(task_name="train")

    # Parse arguments
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

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
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        use_fast=model_args.use_fast_tokenizer,
        padding_side="right"
    )

    # Set seed before initializing model.
    set_seed(training_args.seed)

    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "<pad>"})
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path, 
        torch_dtype=str2dtype(model_args.torch_dtype),
        device_map=model_args.device_map
    )

    # Load training dataset
    train_dataset = SafetyReasoningDataset(
        dataset_name=data_args.dataset_name,
        split="train",
        tokenizer=tokenizer,
        max_length=data_args.max_seq_length,
        # system_inst=data_args.system_inst
        )
    val_dataset = SafetyReasoningDataset(
        dataset_name=data_args.dataset_name,
        split="val",
        tokenizer=tokenizer,
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
    import os
    # Or alternatively, you can set the environment variable
    os.environ["WANDB_MODE"] = "offline"
    os.environ["WANDB_DIR"] = "."
    #---wandb---
    main()
