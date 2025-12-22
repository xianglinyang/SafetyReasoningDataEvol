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
from src.data_utils.RobustSCoT_datasets import load_dataset, SafetyReasoningDataset, SafetyDataCollator
from src.train.probe import calculate_losses_with_dataloader
from src.logger.config import setup_logging
from src.logger.train_log import LoggingCallback
from src.utils.train_utils import load_tokenizer_and_model, merge_lora_checkpoint

from src.train.probe import probe_operator_gradient_direction
from src.train.targeted_generation import select_strategies
from src.evol.question_evol import QuestionEvol
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
    
    # Disable automatic checkpoint saving in trainer
    training_args.save_strategy = "no"
    training_args.save_steps = float('inf')
    training_args.save_total_limit = None
    
    for epoch in range(num_outer_epochs):
        logger.info(f"*** Starting outer epoch {epoch + 1}/{num_outer_epochs} ***")

        # -------------------------------- datasets --------------------------------
        benign_dataset, harmful_dataset = load_dataset(data_args.dataset_name)

        # 1. probe
        # Set model to eval mode and clear cache before probing
        model.eval()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        harmful_dataset_with_losses = probe_operator_gradient_direction(model, tokenizer, harmful_dataset)
        
        # Clean up original harmful_dataset to free memory
        del harmful_dataset
        
        # Clear CUDA cache after probing
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()  # Ensure all operations are complete

        # 2. select strategies and hard samples
        harmful_dataset_with_losses = select_strategies(harmful_dataset_with_losses, top_ratio=data_args.mutation_top_ratio)

        # 3. generate mutations
        # select the ones without mask
        questions = [data['question'] for data in harmful_dataset_with_losses if not data['is_mask']]
        answers = [data['answer'] for data in harmful_dataset_with_losses if not data['is_mask']]
        targeted_strategies = [data['selected_strategy'] for data in harmful_dataset_with_losses if not data['is_mask']]
        
        question_evol = QuestionEvol()
        llm_client = load_model(data_args.mutation_llm)
        mutation_num = data_args.mutation_num
        alpha = data_args.mutation_alpha
        demo_selected_strategy = data_args.demo_selected_strategy
        instruction_variants = await question_evol.generate_prompt_variants_batch_targeted(questions, llm_client, targeted_strategies, num=mutation_num, alpha=alpha, demo_selected_strategy=demo_selected_strategy)
        
        mutation_candidates = dict()
        for question, mutations in instruction_variants.items():
            mutation_candidates[question] = [mutation['text'] for mutation in mutations]

        # 4. compute the losses of the mutations and select the mutations with the highest losses
        # zip the mutations and answers
        
        mutation_questions = []
        mutation_answers = []
        for (question, mutations), answer in zip(mutation_candidates.items(), answers):
            mutation_questions.extend(mutations)
            mutation_answers.extend([answer] * len(mutations))
        
        # Set model to eval mode and clear cache before calculating losses
        model.eval()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        new_losses = calculate_losses_with_dataloader(model, tokenizer, mutation_questions, mutation_answers, device="cuda:0", batch_size=10, max_length=2048)
        
        # Convert tensor to float list and immediately free tensor memory
        new_losses_list = new_losses.cpu().numpy().tolist()
        del new_losses
        
        # Clear CUDA cache after loss calculation
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        losses_map = dict()
        for mutation_question, loss in zip(mutation_questions, new_losses_list):
            losses_map[mutation_question] = loss
        
        # Clean up large lists to free memory
        del mutation_questions, mutation_answers, new_losses_list
        
        for data in harmful_dataset_with_losses:
            if not data['is_mask']:
                question = data['question']
                mutations = mutation_candidates[question]
                losses = [losses_map[mutation] for mutation in mutations]
                data['candidates'] = mutations
                data['losses'] = losses
                data['selected_mutation'] = mutations[np.argmax(losses)]
        
        # Clean up mutation_candidates and losses_map to free memory
        del mutation_candidates, losses_map

        # Set model to eval mode and clear cache before calculating losses
        model.eval()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
                
        # 5. prepare for training by splitting into train and val
        
        # save harmful_dataset_with_losses to json
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        with open(f'{os.path.join(training_args.dataset_log_dir, f"{data_args.dataset_name}_{model_args.model_nickname}_{epoch}_{timestamp}.json")}', 'w') as f:
            json.dump(harmful_dataset_with_losses, f)

        # Add data_type field to datasets before shuffling
        # Mark benign data
        for data in benign_dataset:
            data['data_type'] = 'benign'
        
        # Mark harmful data
        for data in harmful_dataset_with_losses:
            data['data_type'] = 'harmful'

        # shuffle the dataset
        random.shuffle(harmful_dataset_with_losses)
        random.shuffle(benign_dataset)
        train_benign_dataset = benign_dataset[:int(len(benign_dataset)*0.995)]
        train_harmful_dataset = harmful_dataset_with_losses[:int(len(harmful_dataset_with_losses)*0.995)]
        val_benign_dataset = benign_dataset[int(len(benign_dataset)*0.995):]
        val_harmful_dataset = harmful_dataset_with_losses[int(len(harmful_dataset_with_losses)*0.995):]

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
        logger.info(f"*** Starting training for epoch {epoch + 1} ***")
        train_result = trainer.train()

        # -------------------------------- save checkpoint --------------------------------
        # Save checkpoint for this epoch to avoid overwriting
        # Create checkpoint directory in the format expected by merge_lora_checkpoint
        checkpoint_dir = os.path.join(original_output_dir, f"checkpoint-epoch-{epoch}")
        adapter_dir = os.path.join(checkpoint_dir, "adapter_model")
        tokenizer_dir = os.path.join(checkpoint_dir, "tokenizer")
        
        os.makedirs(adapter_dir, exist_ok=True)
        os.makedirs(tokenizer_dir, exist_ok=True)
        
        logger.info(f"*** Saving checkpoint for epoch {epoch + 1} to {checkpoint_dir} ***")
        
        # Save the model (LoRA adapter) and tokenizer in the expected structure
        model.save_pretrained(adapter_dir, safe_serialization=True)
        tokenizer.save_pretrained(tokenizer_dir, safe_serialization=True)

        
        logger.info(f"Checkpoint saved to {checkpoint_dir}")
    
    # -------------------------------- final model merging --------------------------------
    # After all epochs, merge all LoRA checkpoints if using LoRA
    if model_args.lora:
        logger.info("*** Merging all LoRA checkpoints ***")
        merge_lora_checkpoint(model_args.model_name_or_path, original_output_dir)
        logger.info("All LoRA checkpoints merged successfully")

    # TODO
    # Cleanup distributed training
    if dist.is_initialized():
        dist.destroy_process_group()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    # TODO: remove wandb
    #---wandb---
    import os
    # Or alternatively, you can set the environment variable
    os.environ["WANDB_MODE"] = "offline"
    os.environ["WANDB_DIR"] = "."
    #---wandb---
    asyncio.run(main())
