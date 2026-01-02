import os
import json
import torch
import gc
from transformers import AutoModelForCausalLM

def save_model_and_tokenizer(model_name_or_path, model, tokenizer, drop_layers_after, output_dir, trainer):
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n\nModel and tokenizer saving to {output_dir}\n\n")
    
    # merge lora
    print("Merging LoRA weights...")
    merged_model = model.merge_and_unload()
    
    # Free up GPU memory before loading anchor model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
    
    # merge original layers
    if drop_layers_after is not None:
        print(f"Loading anchor model to restore layers after {drop_layers_after}...")
        # Load to CPU first to avoid GPU OOM
        anchor_model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path, 
            torch_dtype=merged_model.dtype, 
            device_map="cpu",  # Load to CPU to avoid OOM
            low_cpu_mem_usage=True
        )
        
        # Merge the layers
        print("Merging truncated model with anchor model layers...")
        merged_model.model.layers = merged_model.model.layers + anchor_model.model.layers[drop_layers_after+1:]
        merged_model.config = anchor_model.config
        
        # Delete anchor model to free memory
        del anchor_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    print("Saving merged model...")
    merged_model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    lorra_config_path = os.path.join(output_dir, "lorra_config.json")
    with open(lorra_config_path, "w", encoding="utf-8") as file:
        json.dump(trainer.lorra_args.to_dict(), file, indent=2)
    
    print("Model and tokenizer saved successfully!")
    
    torch.use_deterministic_algorithms(False)
    if trainer.training_args.do_eval:
        print("Running evaluation...")
        trainer.evaluate()
    