import torch
from torch.utils.data import DataLoader
from accelerate import Accelerator
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    get_linear_schedule_with_warmup,
)
from peft import LoraConfig, get_peft_model
from get_arguments import (
    ModelArguments,
    DataArguments,
    TrainingArguments,
    LoraArguments,
    OrthSAMArguments,
)
from transformers import HfArgumentParser
from ort_train_dataset import ORTDataset, data_reader, ort_collate_fn
from utils import save_tokenizer_and_model
import logging
from tqdm import tqdm
from ort_opti import _clone_grads, OrthSAM, compute_mean_utility_grad, normalize_grad_list, _orth_perturb_first_step
import os
import copy

logger = logging.getLogger(__name__)


# --- Main Function ---
def main():
    # Stage 0: Get the arguments
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments, LoraArguments, OrthSAMArguments))
    model_args, data_args, training_args, lora_args, ort_args = parser.parse_args_into_dataclasses()

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

    # ORT
    optimizer = OrthSAM(
        trainable_params,
        torch.optim.AdamW,
        rho=ort_args.rho,
        adaptive=ort_args.adaptive,
        lr=training_args.lr,
        weight_decay=training_args.weight_decay,
        betas=(0.9, 0.95),
        eps=1e-8,
    )

    # Stage 1: Prepare the dataset
    # prob = 0.5 if "Llama" in model_args.model_name_or_path else 1.0
    prob = 1.0
    ultrachat_dataset = data_reader("ultrachat", prob)
    xstest_dataset = data_reader("xstest", prob)
    rr_dataset = data_reader("circuitbreaker-train-retain", prob)
    
    # Check if use_refusal_retain attribute exists
    retain_dataset = ultrachat_dataset
    refusal_dataset = rr_dataset+xstest_dataset
    
    # refusal / harmful loader
    refusal_train = ORTDataset(refusal_dataset, tokenizer, max_length=data_args.max_length)
    refusal_loader = DataLoader(
        refusal_train,
        batch_size=training_args.per_device_train_batch_size,
        shuffle=True,
        collate_fn=ort_collate_fn,
    )

    # retain / helpful loader
    retain_train = ORTDataset(retain_dataset, tokenizer, max_length=data_args.max_length)
    retain_loader = DataLoader(
        retain_train,
        batch_size=training_args.per_device_train_batch_size,
        shuffle=True,
        collate_fn=ort_collate_fn,
    )

    print("REFUSAL LEN: ", len(refusal_train))
    print("RETAIN LEN: ", len(retain_train))

    # Stage 2: Train the model
    num_epochs = int(training_args.num_train_epochs) if hasattr(training_args, 'num_train_epochs') else getattr(training_args, 'num_epochs', 1)
    total_steps = num_epochs * len(refusal_loader)

    warmup_steps = int(total_steps * training_args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer.base_optimizer, warmup_steps, total_steps)

    # # model, train_loader, scheduler = accelerator.prepare(model, train_loader, scheduler)
    model, refusal_loader, retain_loader, scheduler = accelerator.prepare(model, refusal_loader, retain_loader, scheduler)

    # 训练前算一次全局 mean retain gradient（建议先 256 个 batch 快速试）
    retain_mean_batches = getattr(ort_args, "retain_mean_batches", 512)
    g_u_mean = compute_mean_utility_grad(
        model=model,
        retain_loader=retain_loader,
        trainable_params=trainable_params,
        accelerator=accelerator,
        max_batches=retain_mean_batches,
        use_eval=True,
    )
    g_u_mean = normalize_grad_list(g_u_mean)  # 可选：让投影更稳定

    model.train()
    global_step = 0

    # hyperparameter init
    proj_scale = float(getattr(ort_args, "proj_scale", 1.0))
    one_sided  = bool(getattr(ort_args, "one_sided", True))
    eps = float(getattr(ort_args, "eps", 1e-12))
    lam_u = float(getattr(ort_args, "lam_u", 0.1))  # retain loss weight（可后续做动态）

    # Create progress bar
    pbar = tqdm(total=total_steps, desc="Training", disable=not accelerator.is_main_process)
    for epoch in range(num_epochs):
        retain_iter = iter(retain_loader)
        epoch_loss = 0.0
        epoch_loss_sam = 0.0
        num_batches = 0

        for refusal_batch in refusal_loader:
            # 取 retain batch
            try:
                retain_batch = next(retain_iter)
            except StopIteration:
                retain_iter = iter(retain_loader)
                retain_batch = next(retain_iter)

            model.train()
            optimizer.zero_grad(set_to_none=True)

            # (1) retain normal grad at w: accumulate grads onto existing grads
            loss_u = model(**retain_batch).loss
            loss_u_w = lam_u * loss_u
            accelerator.backward(loss_u_w)

            g_u_step = _clone_grads(trainable_params)   # ✅ 之后加回 step-2 梯度里
            optimizer.zero_grad(set_to_none=True)

            # (2) refusal grad at w
            loss_h = model(**refusal_batch).loss
            accelerator.backward(loss_h)
            g_h = _clone_grads(trainable_params)        # refusal grads (w)
            
            # (3) set g_u_mean as the projection basis for soft projection：g_dir = g_h - proj_scale * proj_{g_u_mean}(g_h)
            alpha, alpha_eff = _orth_perturb_first_step(
                optimizer=optimizer,
                params=trainable_params,
                g_u=g_u_mean,          # ✅ 这里用全局 mean，不再用 batch g_u
                g_h=g_h,
                rho=ort_args.rho,
                eps=eps,
                zero_grad=True,
                fallback="gh",
                proj_scale=proj_scale,
                one_sided=one_sided,
            )

            # (4) refusal SAM step-2: grad at w+e
            loss2 = model(**refusal_batch).loss
            accelerator.backward(loss2)
            # restore weights back to w (keep step-2 grads)
            optimizer.restore()

            # (5) add retain grads back: grad_total = grad_step2 + g_u_step
            with torch.no_grad():
                for p, gu in zip(trainable_params, g_u_step):
                    if gu is None:
                        continue
                    if p.grad is None:
                        p.grad = gu.clone()
                    else:
                        p.grad.add_(gu)
            
            # (6) update once
            if ort_args.orth_sam_max_grad_norm is not None and accelerator.sync_gradients:
                accelerator.clip_grad_norm_(model.parameters(), ort_args.orth_sam_max_grad_norm)

            optimizer.step_base(zero_grad=True)
            scheduler.step()
            global_step += 1

            # (7) logging / progress
            #   loss_refusal      = harmful/refusal loss at w
            #   loss_retain_w     = lam_u * retain loss at w
            #   loss_refusal_sam  = refusal loss at w+e (orth direction)
            loss_h_val = float(loss_h.detach().item())
            loss2_val  = float(loss2.detach().item())   # orth-SAM step-2 loss
            epoch_loss += float(loss_h.detach().item()) + float(loss2.detach().item())
            num_batches += 1

            pbar.update(1)
            if accelerator.is_main_process:
                pbar.set_postfix({
                    'epoch': epoch + 1,
                    'loss_h': f'{loss_h_val:.4f}',
                    'loss_ort': f'{loss2_val:.4f}',
                    'alpha': f'{alpha:.4f}',
                    'alpha_eff': f'{alpha_eff:.4f}',
                    'lr': f'{scheduler.get_last_lr()[0]:.2e}',
                })

            # 释放 loss 张量和 batch 数据（减少显存峰值）
            del loss_h, loss2, refusal_batch, retain_batch

            # 每 100 步清理一次 CUDA 缓存（可选：可能略影响性能，但能缓解碎片）
            if global_step % 100 == 0:
                torch.cuda.empty_cache()
        
        # Print epoch summary
        if accelerator.is_main_process:
            avg_loss = epoch_loss / num_batches
            avg_loss_sam = epoch_loss_sam / num_batches
            accelerator.print(
                f"\n{'='*60}\n"
                f"Epoch {epoch+1}/{num_epochs} Summary:\n"
                f"  Avg Loss: {avg_loss:.4f}\n"
                f"  Avg OrthSAM Loss: {avg_loss_sam:.4f}\n"
                f"{'='*60}\n"
            )
        
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