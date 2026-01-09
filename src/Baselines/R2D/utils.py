import logging
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import get_peft_model, LoraConfig, TaskType, PeftModel
import torch
import os

from src.utils.dtype_utils import str2dtype

logger = logging.getLogger(__name__)

def load_tokenizer_and_model(model_args):
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        use_fast=model_args.use_fast_tokenizer,
        padding_side="right"
    )
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "<pad>"})

    # Load model. Apply LoRA if needed
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path, 
        torch_dtype=str2dtype(model_args.torch_dtype),
        device_map=model_args.device_map
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

    return tokenizer, model


def merge_lora_checkpoint(base_model_path: str, output_path: str, device: str = "cpu"):
    """
    Loads the base model, applies the LoRA adapter, merges them,
    and saves the standalone merged model.

    Args:
        base_model_path: Path to the original base model.
        lora_adapter_path: Path to the saved LoRA adapter directory
                           (e.g., "./training_output/checkpoint-1000/adapter_model").
        merged_output_path: Path where the final merged model will be saved.
    """
    checkpoint_dirs = os.listdir(output_path)
    # find those dirs with format "checkpoint-*"
    checkpoint_dirs = [d for d in checkpoint_dirs if d.startswith("checkpoint-")]

    for checkpoint_dir in checkpoint_dirs:
        merged_output_path = os.path.join(output_path, checkpoint_dir)
        lora_adapter_path = os.path.join(output_path, checkpoint_dir, "adapter_model")
        logger.info(f"Loading PEFT adapter from: {lora_adapter_path}")

        # --- Load Tokenizer ---
        # Try to load tokenizer from checkpoint, fallback to base model
        tokenizer_path = os.path.join(output_path, checkpoint_dir, "tokenizer")
        
        if os.path.exists(tokenizer_path):
            logger.info(f"Loading tokenizer from: {tokenizer_path}")
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        else:
            logger.warning(f"Warning: Tokenizer not found at {tokenizer_path}. Loading from base model instead.")
            tokenizer = AutoTokenizer.from_pretrained(base_model_path)
        
        # Set pad token if needed
        if tokenizer.pad_token is None:
            logger.info("Adding special pad token to tokenizer.")
            tokenizer.add_special_tokens({"pad_token": "<pad>"})
        
        # Save tokenizer to merged output path
        logger.info(f"Saving tokenizer to: {merged_output_path}")
        tokenizer.save_pretrained(merged_output_path, safe_serialization=True)
        logger.info("Tokenizer saved.")

        logger.info(f"Loading base model from: {base_model_path}")
        # --- Load FRESH Base Model for THIS Checkpoint ---
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            torch_dtype=torch.bfloat16, # Adjust dtype as needed
            device_map=device # Load onto CPU initially
        )
        logger.info("Base model loaded.")

        # --- Resize Embeddings for this specific loaded base model ---
        embedding_size = base_model.get_input_embeddings().weight.shape[0]
        if len(tokenizer) > embedding_size:
            logger.info(f"Resizing token embeddings from {embedding_size} to {len(tokenizer)}")
            base_model.resize_token_embeddings(len(tokenizer))
            logger.info("Token embeddings resized for this base model instance.")
        else:
            logger.info("Token embeddings size matches. No resize needed.")

        # Load the PEFT model - this combines base + adapter
        # is_trainable=False is important if you only want to merge, not continue training
        peft_model = PeftModel.from_pretrained(
            base_model,
            lora_adapter_path,
            is_trainable=False,
            device_map=device # Keep on CPU
        )
        logger.info("PEFT adapter loaded onto base model.")

        logger.info("Merging adapter weights...")
        # Merge the adapter layers into the base model
        # This returns the base model with updated weights
        merged_model = peft_model.merge_and_unload()
        logger.info("Merging complete.")

        logger.info(f"Saving merged model to: {merged_output_path}")
        # Save the merged model using standard transformers save_pretrained
        merged_model.save_pretrained(merged_output_path, safe_serialization=True)
        logger.info("Merged model saved.")

        # release memory
        del peft_model
        del merged_model
        del base_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    logger.info("Process finished.")


def save_tokenizer_and_model(model, tokenizer, training_args):
    # Save final model
    logger.info("*** Saving final model ***")

    # LoRA model, need to merge before saving
    merge_model = model.merge_and_unload()
    assert merge_model.get_input_embeddings().weight.shape[0] == len(tokenizer)
    merge_model.save_pretrained(training_args.output_dir, safe_serialization=True)

    tokenizer.save_pretrained(training_args.output_dir, safe_serialization=True)

    logger.info(f"Model saved to {training_args.output_dir}")