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
from ort_opti import OrthSAM, _dot_list, _norm2_list, _clone_grad_list, _get_d
import os
import copy

logger = logging.getLogger(__name__)


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
    if data_args.dataset_name == "circuitbreaker":
        rr_dataset = data_reader("circuitbreaker-train-retain", prob)
        refusal_dataset = rr_dataset
    elif data_args.dataset_name == "R2D-R1":
        refusal_dataset = data_reader("R2D-R1-harmful", prob)
    else:
        raise ValueError(f"Unknown dataset name: {data_args.dataset_name}")
    
    # refusal / harmful loader
    refusal_train = ORTDataset(refusal_dataset, tokenizer, max_length=data_args.max_length)
    refusal_loader = DataLoader(
        refusal_train,
        batch_size=training_args.per_device_train_batch_size,
        shuffle=True,
        collate_fn=ort_collate_fn,
    )

    print("REFUSAL LEN: ", len(refusal_train))

    # Stage 2: Train the model
    num_epochs = int(training_args.num_train_epochs) if hasattr(training_args, 'num_train_epochs') else getattr(training_args, 'num_epochs', 1)
    total_steps = num_epochs * len(refusal_loader)

    warmup_steps = int(total_steps * training_args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer.base_optimizer, warmup_steps, total_steps)

    # model, train_loader, scheduler = accelerator.prepare(model, train_loader, scheduler)
    model, refusal_loader, scheduler = accelerator.prepare(model, refusal_loader, scheduler)

    model.train()
    global_step = 0

    # hyperparameter init
    beta = getattr(ort_args, "ema_beta", 0.97)   # retain grad EMA

    g_u_ema = None  # list-of-tensors, same structure as trainable_params grads
    eps = getattr(ort_args, "eps", 1e-12)

    # copy initial params for anchor direction
    theta0 = [p.detach().clone() for p in trainable_params]  # anchor: initial params
    one_sided = True
    d_ema = None

    proj_scale = float(getattr(ort_args, "proj_scale", 1.0))
    # Create progress bar
    pbar = tqdm(total=total_steps, desc="Training", disable=not accelerator.is_main_process)
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        num_batches = 0

        for refusal_batch in refusal_loader:
            model.train()

            # (0) build anchor direction d = theta - theta0 (or EMA)
            d, d_ema = _get_d(trainable_params, theta0, d_ema, use_ema=True, beta=beta)
            dd = _norm2_list(d)
            d_norm = float(torch.sqrt((dd if dd is not None else torch.tensor(0.0, device=trainable_params[0].device)) + eps).item())

            optimizer.zero_grad(set_to_none=True)

            # ------------------------------------------------------------
            # (1) refusal step-1 grad g_h at w
            # ------------------------------------------------------------
            loss_h = model(**refusal_batch).loss
            accelerator.backward(loss_h)
            g_h = _clone_grad_list(trainable_params)  # list of tensors

            # ------------------------------------------------------------
            # (2) project out component along d: g_dir = g_h - proj_d(g_h)
            #     alpha = <g_h, d> / <d, d>
            # ------------------------------------------------------------
            alpha = 0.0
            if dd is None or dd.detach().item() == 0.0:
                g_dir = g_h
            else:
                gh_dot_d = _dot_list(g_h, d)
                coef = gh_dot_d / (dd + eps)
                alpha = float(coef.detach().item())

                if (not one_sided) or (alpha > 0.0):
                    g_dir = []
                    for gh_i, d_i in zip(g_h, d):
                        if gh_i is None:
                            g_dir.append(None)
                        else:
                            g_dir.append(gh_i - proj_scale * coef * d_i)
                else:
                    g_dir = g_h

            # ------------------------------------------------------------
            # (3) custom SAM first-step using g_dir, store old_p so optimizer.restore() works
            #     w <- w + rho * g_dir / ||g_dir||
            # ------------------------------------------------------------
            with torch.no_grad():
                gdir_n2 = _norm2_list(g_dir)
                gdir_norm = torch.sqrt((gdir_n2 if gdir_n2 is not None else torch.tensor(0.0, device=loss_h.device)) + eps)

                # fallback: if orth direction vanishes, use original g_h
                if gdir_norm.detach().item() == 0.0:
                    g_dir = g_h
                    gdir_n2 = _norm2_list(g_dir)
                    gdir_norm = torch.sqrt((gdir_n2 if gdir_n2 is not None else torch.tensor(0.0, device=loss_h.device)) + eps)

                scale = ort_args.rho / (gdir_norm + eps)

                for p, step_dir in zip(trainable_params, g_dir):
                    if step_dir is None:
                        continue
                    optimizer.state[p]["old_p"] = p.detach().clone()
                    p.add_(step_dir, alpha=scale)

            # clear step-1 grads (SAM standard)
            optimizer.zero_grad(set_to_none=True)

            # ------------------------------------------------------------
            # (4) refusal step-2 at w+e: compute grads
            # ------------------------------------------------------------
            loss2 = model(**refusal_batch).loss
            accelerator.backward(loss2)

            # restore to w (keep grads from step-2)
            optimizer.restore()

            # ------------------------------------------------------------
            # (5) clip + one update
            # ------------------------------------------------------------
            if ort_args.orth_sam_max_grad_norm is not None and accelerator.sync_gradients:
                accelerator.clip_grad_norm_(model.parameters(), ort_args.orth_sam_max_grad_norm)

            optimizer.step_base(zero_grad=True)
            scheduler.step()
            global_step += 1

            # -----------------------------
            # logging (按你习惯的字段)
            # -----------------------------
            loss_h_val = float(loss_h.detach().item())
            loss2_val  = float(loss2.detach().item())

            loss_report = loss_h.detach() + loss2.detach()
            epoch_loss += float(loss_report.item())
            num_batches += 1

            pbar.update(1)
            if accelerator.is_main_process:
                pbar.set_postfix({
                    'epoch': epoch + 1,
                    'loss_h': f'{loss_h_val:.4f}',
                    'loss_ort': f'{loss2_val:.4f}',
                })
            del loss_h, loss2, refusal_batch
            if global_step % 100 == 0:
                torch.cuda.empty_cache()

        if accelerator.is_main_process and global_step % 20 == 0:
            avg_loss = epoch_loss / num_batches
            accelerator.print(
                f"step={global_step}/{total_steps} "
                f"loss_h={loss_h_val:.4f} "
                f"loss_ort={loss2_val:.4f} "
                f"avg_loss={avg_loss:.4f} "
                f"lr={scheduler.get_last_lr()[0]:.2e}"
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