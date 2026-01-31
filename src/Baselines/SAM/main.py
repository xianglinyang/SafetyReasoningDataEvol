import torch
from torch.utils.data import DataLoader
from accelerate import Accelerator
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    get_linear_schedule_with_warmup,
)
from peft import LoraConfig, get_peft_model
from sam_opti import SAM
from get_arguments import (
    ModelArguments,
    DataArguments,
    TrainingArguments,
    LoraArguments,
    SAMArguments,
)
from transformers import HfArgumentParser
from sam_train_dataset import SAMDataset, data_reader, sam_collate_fn
from utils import save_tokenizer_and_model
import logging
from tqdm import tqdm
import os
import copy

logger = logging.getLogger(__name__)


def main():
    # Stage 0: Get the arguments
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments, LoraArguments, SAMArguments))
    model_args, data_args, training_args, lora_args, sam_args = parser.parse_args_into_dataclasses()

    # Model parameters
    accelerator = Accelerator(mixed_precision="bf16")
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        torch_dtype=torch.bfloat16,
        device_map=None,  # 交给 accelerate.prepare
    )
    
    # 启用 gradient checkpointing 以节省内存
    if hasattr(model, 'enable_input_require_grads'):
        model.enable_input_require_grads()
    if hasattr(model, 'gradient_checkpointing_enable'):
        model.gradient_checkpointing_enable()

    # LoRA
    lora_config = LoraConfig(
        **lora_args.__dict__
    )
    model = get_peft_model(model, lora_config, autocast_adapter_dtype=False)
    model.print_trainable_parameters()
    trainable_params = [p for p in model.parameters() if p.requires_grad]

    # SAM
    optimizer = SAM(
        trainable_params,
        torch.optim.AdamW,
        rho=sam_args.rho,
        adaptive=sam_args.adaptive,
        lr=training_args.lr,
        weight_decay=training_args.weight_decay,
        betas=(0.9, 0.95),
        eps=1e-8,
    )

    # Stage 1: Prepare the dataset (similar to ORT setup)
    prob = 0.5 if "Llama" in model_args.model_name_or_path else 1.0
    ultrachat_dataset = data_reader("ultrachat", prob)
    xstest_dataset = data_reader("xstest", prob)
    rr_dataset = data_reader("circuitbreaker-train-retain", prob)
    
    # Separate refusal (harmful) and retain (helpful) datasets
    retain_dataset = ultrachat_dataset
    refusal_dataset = rr_dataset+xstest_dataset
    
    
    # refusal / harmful loader
    refusal_train = SAMDataset(refusal_dataset, tokenizer, max_length=data_args.max_length)
    refusal_loader = DataLoader(
        refusal_train,
        batch_size=training_args.per_device_train_batch_size,
        shuffle=True,
        collate_fn=sam_collate_fn,
    )

    # retain / helpful loader
    retain_train = SAMDataset(retain_dataset, tokenizer, max_length=data_args.max_length)
    retain_loader = DataLoader(
        retain_train,
        batch_size=training_args.per_device_train_batch_size,
        shuffle=True,
        collate_fn=sam_collate_fn,
    )

    print("REFUSAL LEN: ", len(refusal_train))
    print("RETAIN LEN: ", len(retain_train))

    # Stage 2: Train the model
    num_epochs = int(training_args.num_train_epochs) if hasattr(training_args, 'num_train_epochs') else getattr(training_args, 'num_epochs', 1)
    total_steps = num_epochs * len(refusal_loader)
    
    warmup_steps = int(total_steps * training_args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer.base_optimizer, warmup_steps, total_steps)

    # Don't prepare optimizer with accelerator to keep SAM methods accessible
    model, refusal_loader, retain_loader, scheduler = accelerator.prepare(model, refusal_loader, retain_loader, scheduler)
    
    model.train()
    global_step = 0

    pbar = tqdm(total=total_steps, desc="Training", disable=not accelerator.is_main_process)
    
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        epoch_loss_sam = 0.0
        num_batches = 0
        retain_iter = iter(retain_loader)
        
        for refusal_batch in refusal_loader:
            try:
                retain_batch = next(retain_iter)
            except StopIteration:
                retain_iter = iter(retain_loader)
                retain_batch = next(retain_iter)

            model.train()
            optimizer.zero_grad(set_to_none=True)

            # -----------------------------
            # (A) refusal SAM step-1 (at w): get grad then perturb to w+e
            # -----------------------------
            loss_refusal = model(**refusal_batch).loss
            accelerator.backward(loss_refusal)
            optimizer.first_step(zero_grad=True)   # w -> w+e, clear grads

            # -----------------------------
            # (B) refusal SAM step-2 (at w+e): compute grad at perturbed weights
            # -----------------------------
            loss_refusal_sam = model(**refusal_batch).loss
            accelerator.backward(loss_refusal_sam)

            # restore back to w (DO NOT clear grads)
            optimizer.restore()

            # -----------------------------
            # (C) retain normal loss (at w): accumulate grads onto existing grads
            # -----------------------------
            loss_retain = model(**retain_batch).loss
            loss_retain_w = sam_args.lam_u * loss_retain
            accelerator.backward(loss_retain_w)

            # -----------------------------
            # (D) clip + ONE final update
            # -----------------------------
            if sam_args.sam_max_grad_norm is not None and accelerator.sync_gradients:
                accelerator.clip_grad_norm_(model.parameters(), sam_args.sam_max_grad_norm)

            optimizer.step_base(zero_grad=True)
            scheduler.step()
            global_step += 1

            # -----------------------------
            # (E) logging
            # -----------------------------
            loss_report = loss_retain_w.detach() + loss_refusal.detach() + loss_refusal_sam.detach()

            epoch_loss += float(loss_report)
            epoch_loss_sam += float(loss_refusal_sam.detach())
            num_batches += 1

            pbar.update(1)
            if accelerator.is_main_process:
                pbar.set_postfix({
                    "epoch": epoch + 1,
                    "loss_total": f"{loss_report.item():.4f}",
                    "retain": f"{loss_retain_w.item():.4f}",
                    "ref1": f"{loss_refusal.item():.4f}",
                    "sam": f"{loss_refusal_sam.item():.4f}",
                })

        # End of epoch logging
        if accelerator.is_main_process and num_batches > 0:
            avg_epoch_loss = epoch_loss / num_batches
            avg_epoch_loss_sam = epoch_loss_sam / num_batches
            logger.info(f"Epoch {epoch + 1}/{num_epochs} - Avg Loss: {avg_epoch_loss:.4f}, Avg SAM Loss: {avg_epoch_loss_sam:.4f}")
        
        # Save checkpoint after each epoch
        if accelerator.is_main_process:
            epoch_output_dir = os.path.join(training_args.output_dir, f"checkpoint-epoch-{epoch + 1}")
            os.makedirs(epoch_output_dir, exist_ok=True)
            logger.info(f"*** Saving checkpoint for epoch {epoch + 1} to {epoch_output_dir} ***")
            
            # Create temporary training args for this checkpoint
            epoch_training_args = copy.deepcopy(training_args)
            epoch_training_args.output_dir = epoch_output_dir
            
            unwrapped_model = accelerator.unwrap_model(model)
            save_tokenizer_and_model(unwrapped_model, tokenizer, epoch_training_args)
            logger.info(f"Checkpoint saved for epoch {epoch + 1}")
    
    pbar.close()

    # Stage 3: Save the final model
    if accelerator.is_main_process:
        logger.info("*** Saving final model and tokenizer ***")
        unwrapped_model = accelerator.unwrap_model(model)
        save_tokenizer_and_model(unwrapped_model, tokenizer, training_args)


if __name__ == "__main__":
    main()
