import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from functools import partial
from src.Baselines.circuitbreaker.get_arguments import (
    ModelArguments,
    TrainingArguments, 
    LoraArguments, 
    LorraArguments,
)
from src.Baselines.circuitbreaker.cb_train_dataset import (
    CircuitBreakerDataset,
    data_reader,
)
from src.Baselines.circuitbreaker.utils import (
    save_model_and_tokenizer,
)
from src.Baselines.circuitbreaker.cb_trainer import (
    CircuitBreakerTrainer,
    data_collator
)
from peft import LoraConfig, get_peft_model
from transformers.integrations import deepspeed
import torch
import torch.distributed as dist
import numpy as np
import atexit
import logging

logger = logging.getLogger(__name__)


def main():
    parser = transformers.HfArgumentParser(
        (ModelArguments, TrainingArguments, LoraArguments, LorraArguments)
    )
    (
        model_args,
        training_args,
        lora_args,
        lorra_args,
    ) = parser.parse_args_into_dataclasses()

    print(lorra_args.to_dict())
    print(lora_args)
    print(model_args)
    print(training_args)

    # Don't use device_map='auto' in distributed training mode
    # Check if we're in distributed mode
    is_distributed = training_args.local_rank != -1 or training_args.world_size > 1
    device_map = None if is_distributed else "auto"
    
    if len(training_args.fsdp) > 0 or deepspeed.is_deepspeed_zero3_enabled():
        logging.warning(
            "FSDP and ZeRO3 are both currently incompatible with QLoRA."
        )

    model_name_or_path = model_args.model_name_or_path
    target_layers = lorra_args.target_layers
    transform_layers = lorra_args.transform_layers
    full_layers = lorra_args.full_layers

    # define updated layers
    lorra_target_layers = [int(layer) for layer in target_layers.split(",")] # target representations
    if "-1" in transform_layers:
        lora_layers_to_transform = [i for i in range(max(lorra_target_layers) + 1)]
    else:
        lora_layers_to_transform = [int(layer) for layer in transform_layers.split(",")] # transform representations

    lora_config = LoraConfig(
        r=lora_args.lora_r,
        lora_alpha=lora_args.lora_alpha,
        target_modules=lora_args.lora_target_modules,
        lora_dropout=lora_args.lora_dropout,
        bias=lora_args.lora_bias,
        layers_to_transform=lora_layers_to_transform,
        task_type="CAUSAL_LM",
    )

    drop_layers_after = max(lorra_target_layers) if not full_layers else None
    print("lorra_transform_layers", lora_layers_to_transform)
    print("drop_layers_after", drop_layers_after)

    config = AutoConfig.from_pretrained(model_name_or_path)
    if drop_layers_after:
        config.num_hidden_layers = drop_layers_after+1

    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="left",
        use_fast="LlamaForCausalLM" not in config.architectures,
    )
    tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
    extra_save_kargs = dict(tokenizer=tokenizer)
    save_model_function = save_model_and_tokenizer
    
    # Build model loading arguments
    # model_name_or_path should be passed as first positional argument, not as keyword
    model_load_kwargs = {
        "config": config,
        "cache_dir": training_args.cache_dir,
    }
    # Only add device_map if not None (i.e., not in distributed mode)
    if device_map is not None:
        model_load_kwargs["device_map"] = device_map
    
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, **model_load_kwargs)
    save_model_function = partial(save_model_function, 
                    model_name_or_path=model_name_or_path, 
                    drop_layers_after=drop_layers_after, 
                    output_dir=training_args.output_dir,
                    **extra_save_kargs)

    print(lora_args.lora_target_modules, lora_layers_to_transform)

    model = get_peft_model(model, lora_config)
    print("model", model)


    if training_args.deepspeed is not None and training_args.local_rank == 0:
        model.print_trainable_parameters()

    if training_args.gradient_checkpointing:
        model.enable_input_require_grads()

    ultrachat_dataset = data_reader("ultrachat")
    xstest_dataset = data_reader("xstest")
    circuitbreaker_train_cb_dataset = data_reader("circuitbreaker-train-cb")
    circuitbreaker_val_dataset = data_reader("circuitbreaker-val")

    retain_dataset = ultrachat_dataset + xstest_dataset
    refusal_dataset = circuitbreaker_train_cb_dataset
    val_dataset = circuitbreaker_val_dataset

    train_dataset = CircuitBreakerDataset(refusal_dataset, retain_dataset, tokenizer, model_name_or_path, max_length=1024)
    val_dataset = CircuitBreakerDataset(refusal_dataset, retain_dataset, tokenizer, model_name_or_path, max_length=1024)

    print("TRAIN LEN: ", len(train_dataset))
    print("VAL LEN: ", len(val_dataset))

    training_args.remove_unused_columns = False

    trainer = CircuitBreakerTrainer(
        model=model, 
        tokenizer=tokenizer, 
        args=training_args, 
        train_dataset=train_dataset, 
        eval_dataset=val_dataset,
        data_collator=data_collator,
        lorra_args=lorra_args,
        training_args=training_args,
        lorra_target_layers=lorra_target_layers,
    )
    model.config.use_cache = False
    
    # Only register atexit handler on rank 0 to avoid conflicts in distributed training
    if training_args.local_rank == -1 or training_args.local_rank == 0:
        atexit.register(save_model_function, model=model, trainer=trainer)
    
    try:
        trainer.train()
    finally:
        # Cleanup distributed training
        if dist.is_initialized():
            dist.destroy_process_group()
        
        # Clear CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
if __name__ == "__main__":
    import sys
    import traceback
    
    SEED = 42
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.use_deterministic_algorithms(True)

    try:
        main()
    except Exception as e:
        # Print full traceback to help debug
        print(f"\n{'='*80}")
        print("ERROR: Training failed with the following exception:")
        print(f"{'='*80}")
        traceback.print_exc()
        print(f"{'='*80}\n")
        sys.exit(1)