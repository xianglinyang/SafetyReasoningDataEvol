import torch
from torch.utils.data import DataLoader
from accelerate import Accelerator
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    get_linear_schedule_with_warmup,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from ort_opti import ORT
from get_arguments import (
    ModelArguments,
    DataArguments,
    TrainingArguments,
    LoraArguments,
    ORTArguments,
)
from transformers import HfArgumentParser
from ort_train_dataset import ORTDataset, data_reader, ort_collate_fn
from utils import save_tokenizer_and_model
import logging
from tqdm import tqdm
from ort_opti import set_projected_grads

logger = logging.getLogger(__name__)


def main():
    # Stage 0: Get the arguments
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments, LoraArguments, ORTArguments))
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

    # LoRA
    lora_config = LoraConfig(
        **lora_args.__dict__
    )
    model = get_peft_model(model, lora_config, autocast_adapter_dtype=False)
    model.print_trainable_parameters()
    trainable_params = [p for p in model.parameters() if p.requires_grad]

    # ORT
    optimizer = ORT(
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
    prob = 0.5 if "Llama" in model_args.model_name_or_path else 1.0
    ultrachat_dataset = data_reader("ultrachat", prob)
    xstest_dataset = data_reader("xstest", prob)
    
    # Check if use_refusal_retain attribute exists
    retain_dataset = ultrachat_dataset + xstest_dataset
    refusal_dataset = data_reader("circuitbreaker-train-retain", prob)
    
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

    print("TRAIN LEN: ", len(train_dataset))
    print("VAL LEN: ", len(val_dataset))

    train_loader = DataLoader(train_dataset, batch_size=training_args.per_device_train_batch_size, shuffle=True, collate_fn=ort_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=training_args.per_device_train_batch_size, shuffle=False, collate_fn=ort_collate_fn)

    # Stage 2: Train the model
    num_epochs = int(training_args.num_train_epochs) if hasattr(training_args, 'num_train_epochs') else getattr(training_args, 'num_epochs', 1)
    total_steps = num_epochs * len(refusal_loader)

    warmup_steps = int(total_steps * training_args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer.base_optimizer, warmup_steps, total_steps)

    # model, train_loader, scheduler = accelerator.prepare(model, train_loader, scheduler)
    model, refusal_loader, retain_loader, scheduler = accelerator.prepare(model, refusal_loader, retain_loader, scheduler)

    model.train()
    global_step = 0
    
    # Create progress bar
    pbar = tqdm(total=total_steps, desc="Training", disable=not accelerator.is_main_process)
    retain_iter = iter(retain_loader)

    for refusal_batch in refusal_loader:
        # 取一批 utility/retain
        try:
            retain_batch = next(retain_iter)
        except StopIteration:
            retain_iter = iter(retain_loader)
            retain_batch = next(retain_iter)

        # ========= (A) 计算 g_h on refusal =========
        optimizer.zero_grad(set_to_none=True)
        loss_h = model(**refusal_batch).loss
        accelerator.backward(loss_h)
        gh = [p.grad.detach().clone() if p.grad is not None else None for p in trainable_params]

        # ========= (B) 计算 g_u on retain =========
        optimizer.zero_grad(set_to_none=True)
        loss_u = model(**retain_batch).loss
        accelerator.backward(loss_u)
        gu = [p.grad.detach().clone() if p.grad is not None else None for p in trainable_params]

        # ========= (C) 写入投影后的梯度到 p.grad，然后做 ORT 的 first_step =========
        optimizer.zero_grad(set_to_none=True)
        alpha = set_projected_grads(trainable_params, gh, gu, eps=1e-12)

        accelerator.clip_grad_norm_(trainable_params, training_args.max_grad_norm)
        optimizer.first_step(zero_grad=True)

        # ========= (D) 2nd forward/backward on (w+eps) =========
        # outer loss：这里先用 refusal（harmful）batch 作为最直接的 baseline
        # loss2 = model(**refusal_batch).loss
        loss2 = model(**refusal_batch).loss + lam_u * model(**retain_batch).loss

        accelerator.backward(loss2)

        accelerator.clip_grad_norm_(trainable_params, training_args.max_grad_norm)
        optimizer.second_step(zero_grad=True)
        scheduler.step()

        global_step += 1
        pbar.update(1)
        pbar.set_postfix({
            'epoch': epoch + 1,
            'loss_h': f'{loss_h.item():.4f}',
            'loss_u': f'{loss_u.item():.4f}',
            'loss_ort': f'{loss2.item():.4f}',
            'alpha': f'{alpha:.3e}',
            'lr': f'{scheduler.get_last_lr()[0]:.2e}'
        })

    
    pbar.close()

    # Stage 3: Save the model
    logger.info("*** Saving final model and tokenizer ***")
    save_tokenizer_and_model(model, tokenizer, training_args)


if __name__ == "__main__":
    main()