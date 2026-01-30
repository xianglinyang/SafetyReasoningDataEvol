import torch
from torch.utils.data import DataLoader
from accelerate import Accelerator
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    get_linear_schedule_with_warmup,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
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

    # Stage 1: Prepare the dataset
    prob = 0.5 if "Llama" in model_args.model_name_or_path else 1.0
    ultrachat_dataset = data_reader("ultrachat", prob)
    xstest_dataset = data_reader("xstest", prob)
    
    # Check if use_refusal_retain attribute exists
    if prob == 0.5:
        retain_dataset = data_reader("circuitbreaker-train-retain", prob)
        retain_dataset = retain_dataset + xstest_dataset + ultrachat_dataset
    else: 
        retain_dataset = ultrachat_dataset + xstest_dataset
    
    circuitbreaker_val_dataset = data_reader("circuitbreaker-val", prob)

    train_dataset = SAMDataset(retain_dataset, tokenizer, max_length=data_args.max_length)
    val_dataset = SAMDataset(circuitbreaker_val_dataset, tokenizer, max_length=data_args.max_length)

    print("TRAIN LEN: ", len(train_dataset))
    print("VAL LEN: ", len(val_dataset))

    train_loader = DataLoader(train_dataset, batch_size=training_args.per_device_train_batch_size, shuffle=True, collate_fn=sam_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=training_args.per_device_train_batch_size, shuffle=False, collate_fn=sam_collate_fn)

    # Stage 2: Train the model
    num_epochs = int(training_args.num_train_epochs) if hasattr(training_args, 'num_train_epochs') else getattr(training_args, 'num_epochs', 1)
    total_steps = num_epochs * len(train_loader)
    warmup_steps = int(total_steps * training_args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer.base_optimizer, warmup_steps, total_steps)

    # Don't prepare optimizer with accelerator to keep SAM methods accessible
    model, train_loader, scheduler = accelerator.prepare(model, train_loader, scheduler)

    model.train()
    global_step = 0
    
    # Create progress bar
    pbar = tqdm(total=total_steps, desc="Training", disable=not accelerator.is_main_process)
    
    for epoch in range(num_epochs):
        epoch_loss = 0
        epoch_loss_sam = 0
        num_batches = 0
        
        for batch in train_loader:
            # ===== 1st forward/backward =====
            loss = model(**batch).loss
            accelerator.backward(loss)

            accelerator.clip_grad_norm_(trainable_params, training_args.max_grad_norm)
            optimizer.first_step(zero_grad=True)

            # ===== 2nd forward/backward (w+eps) =====
            loss2 = model(**batch).loss
            accelerator.backward(loss2)

            accelerator.clip_grad_norm_(trainable_params, training_args.max_grad_norm)
            optimizer.second_step(zero_grad=True)
            scheduler.step()

            global_step += 1
            num_batches += 1
            epoch_loss += loss.item()
            epoch_loss_sam += loss2.item()
            
            # Update progress bar
            pbar.update(1)
            pbar.set_postfix({
                'epoch': epoch + 1,
                'loss': f'{loss.item():.4f}',
                'loss_sam': f'{loss2.item():.4f}',
                'lr': f'{scheduler.get_last_lr()[0]:.2e}'
            })
            
            if accelerator.is_main_process and global_step % 20 == 0:
                accelerator.print(
                    f"epoch={epoch+1}/{num_epochs} step={global_step}/{total_steps} "
                    f"loss={loss.item():.4f} loss_sam={loss2.item():.4f} "
                    f"lr={scheduler.get_last_lr()[0]:.2e}"
                )
        
        # Print epoch summary
        if accelerator.is_main_process:
            avg_loss = epoch_loss / num_batches
            avg_loss_sam = epoch_loss_sam / num_batches
            accelerator.print(
                f"\n{'='*60}\n"
                f"Epoch {epoch+1}/{num_epochs} Summary:\n"
                f"  Avg Loss: {avg_loss:.4f}\n"
                f"  Avg SAM Loss: {avg_loss_sam:.4f}\n"
                f"{'='*60}\n"
            )
    
    pbar.close()

    # Stage 3: Save the model
    logger.info("*** Saving final model and tokenizer ***")
    save_tokenizer_and_model(model, tokenizer, training_args)


if __name__ == "__main__":
    main()