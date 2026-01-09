from get_arguments import (
    ModelArguments,
    DataArguments,
    TrainingArguments,
    LoraArguments,
)
from r2d_train_dataset import R2DDataset, data_reader
from r2d_trainer import R2DTrainer
from utils import save_tokenizer_and_model, merge_lora_checkpoint
from transformers import HfArgumentParser, AutoTokenizer, AutoModelForCausalLM, DataCollatorForSeq2Seq
from peft import LoraConfig, get_peft_model
import torch
import torch.distributed as dist
import logging
import gc
import os

# Disable bitsandbytes if not needed (to avoid CUDA library errors)
os.environ['BITSANDBYTES_NOWELCOME'] = '1'

logger = logging.getLogger(__name__)

def main():
    # Stage 0: Get the arguments
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments, LoraArguments))
    model_args, data_args, training_args, lora_args = parser.parse_args_into_dataclasses()

    # Stage 1: Expand the tokenizer and model
    logger.info(f"Expanding the tokenizer and model for {model_args.model_name_or_path}")

    model_path = model_args.model_name_or_path

    # Load model with auto device_map for efficient initialization
    # Note: Will be moved by Trainer later for DDP training
    llm = AutoModelForCausalLM.from_pretrained(
        model_path, 
        torch_dtype=torch.bfloat16, 
        low_cpu_mem_usage=True,  # Use less CPU memory during loading
        trust_remote_code=True,
        use_cache=False  # Disable cache when using gradient checkpointing
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    
    tokenizer.pad_token = tokenizer.eos_token
    # Set model_max_length to avoid truncation warnings
    tokenizer.model_max_length = data_args.max_length
    tokenizer.add_tokens(["[SAFE]", "[UNSAFE]", "[RETHINK]", "<think>", "</think>", "[R2D-Reserve-0]", "[R2D-Reserve-1]", "[R2D-Reserve-2]"])

    llm.resize_token_embeddings(len(tokenizer)) 
    logger.info(f"llm.model.embed_tokens.weight[-8:, :]: {llm.model.embed_tokens.weight[-8:, :]}")
    logger.info(f"llm.lm_head.weight[:, -8:]: {llm.lm_head.weight[:, -8:]}")

    llm.config.vocab_size = len(tokenizer)
    
    # Fix for PEFT warning: ensure weight tying is disabled when using LoRA with tied layers
    if hasattr(llm.config, 'tie_word_embeddings'):
        llm.config.tie_word_embeddings = False
        logger.info("Disabled tie_word_embeddings to avoid PEFT complications")
    # llm.save_pretrained(os.path.join(args.output_path, "r2d_outputs", model_name + "-expanded"))
    # tokenizer.save_pretrained(os.path.join(args.output_path, "r2d_outputs", model_name + "-expanded"))

    # Stage 2: Train the model

    logger.info(f"Training/evaluation parameters {training_args}")

    # tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
    # model = AutoModelForCausalLM.from_pretrained(model_args.model_name_or_path, torch_dtype=torch.bfloat16, trust_remote_code=True)

    dataset = data_reader(data_args.dataset_path)

    train_dataset = R2DDataset(
        dataset=dataset,
        tokenizer=tokenizer,
        max_length=data_args.max_length,
        split=data_args.split,
    )

    print(lora_args)
    lora_config = LoraConfig(
        **lora_args.__dict__
    )

    if model_args.use_lora:
        llm.enable_input_require_grads()
        # Set autocast_adapter_dtype=False to avoid weight tying issues
        llm = get_peft_model(llm, lora_config, autocast_adapter_dtype=False)
    
    # Disable use_cache when gradient_checkpointing is enabled
    if training_args.gradient_checkpointing:
        llm.config.use_cache = False
        logger.info("Disabled use_cache for gradient checkpointing")
    
    # Move model to GPU - Trainer will handle multi-GPU distribution
    # Don't specify device_map here, let Trainer/Accelerate handle it
    logger.info("Moving model to GPU for training...")
    
    # Force garbage collection to release CPU memory after model is moved to GPU
    gc.collect()

    trainer = R2DTrainer(
        model=llm,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,  # Pass tokenizer to enable checkpoint saving
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer)
    )
    trainer.train()

    # Stage 3: Save the final model and tokenizer
    logger.info("*** Saving final model and tokenizer ***")
    save_tokenizer_and_model(llm, tokenizer, training_args)

    # Stage 4: Merge LoRA checkpoints (before cleanup for better debugging)
    logger.info("Starting checkpoint merge...")
    merge_lora_checkpoint(model_args.model_name_or_path, training_args.output_dir)

    # Stage 5: Cleanup distributed training
    if dist.is_initialized():
        dist.destroy_process_group()

    # release memory
    del llm  # Fixed: was 'model', should be 'llm'
    del tokenizer
    
    # Force garbage collection and clear caches
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    logger.info("All done! Memory released (GPU and CPU).")

if __name__ == "__main__":
    main()