'''
The training script for the Safety Reasoning Data Evol model with LoRA.
'''

import os
import sys
import logging
import time

import transformers
from transformers import HfArgumentParser, set_seed, AutoTokenizer, AutoModelForCausalLM, Trainer, DataCollatorForLanguageModeling
from peft import LoraConfig, PeftModel, TaskType, get_peft_model

from src.train.get_arguments import ModelArguments, DataArguments, TrainingArguments
from src.train.data_utils.safety_datasets import SafetyReasoningDataset
from src.train.data_utils.model_configs import MODEL_CONFIGS

logger = logging.getLogger(__name__)

# redirect logging from console to file
def setup_logging(training_args):
    """Setup logging to both file and console"""
    
    # Create logs directory if it doesn't exist
    log_dir = os.path.join(training_args.output_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    
    # Create log file path with timestamp
    time_stamp = time.strftime("%Y%m%d-%H%M%S")
    log_file = os.path.join(log_dir, f"training_{time_stamp}.log")
    
    # Setup logging format
    log_format = "%(asctime)s - %(levelname)s - %(name)s - %(message)s"
    date_format = "%m/%d/%Y %H:%M:%S"
    
    # Setup logging to both file and console
    logging.basicConfig(
        format=log_format,
        datefmt=date_format,
        level=logging.INFO,
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    logger.info(f"Logging to {log_file}")

def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Setup logging
    setup_logging(training_args)

    if training_args.should_log:
        transformers.utils.logging.set_verbosity_info()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training parameters {training_args}")
    logger.info(f"Model parameters {model_args}")
    logger.info(f"Dataset parameters {data_args}")

    # # Set seed before initializing model.
    # set_seed(training_args.seed)

    # tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
    # # add_padding_to_tokenizer(tokenizer)
    # model = AutoModelForCausalLM.from_pretrained(model_args.model_name_or_path, torch_dtype=model_args.torch_dtype)
    # # Load training dataset
    # train_dataset = SafetyReasoningDataset(
    #     dataset_name=data_args.dataset_name,
    #     split="train",
    #     tokenizer=tokenizer,
    #     model_name=model_args.model_name_abbr)
    # val_dataset = SafetyReasoningDataset(
    #     dataset_name=data_args.dataset_name,
    #     split="val",
    #     tokenizer=tokenizer,
    #     model_name=model_args.model_name_abbr)

    # # resize embeddings if needed (e.g. for LlamaTokenizer)
    # embedding_size = model.get_input_embeddings().weight.shape[0]
    # if len(tokenizer) > embedding_size:
    #     model.resize_token_embeddings(len(tokenizer))
    #     # if you load lora model and resize the token embeddings, the requires_grad flag is set to True for embeddings
    #     if isinstance(model, PeftModel):
    #         model.get_input_embeddings().weight.requires_grad = False
    #         model.get_output_embeddings().weight.requires_grad = False

    # if not isinstance(model, PeftModel) and model_args.lora:
    #     lora_config = LoraConfig(
    #         task_type=TaskType.CAUSAL_LM,
    #         inference_mode=False,
    #         r=model_args.lora_r,
    #         lora_alpha=model_args.lora_alpha,
    #         lora_dropout=model_args.lora_dropout,
    #         target_modules=model_args.lora_target_modules,
    #     )
    #     model = get_peft_model(model, lora_config)
    #     logger.info(
    #         f"Applied LoRA to model."
    #     )
    #     model.print_trainable_parameters()

    #     # for checkpointing
    #     if hasattr(model, "enable_input_require_grads"):
    #         model.enable_input_require_grads()
    #     else:
    #         def make_inputs_require_grad(module, input, output):
    #             output.requires_grad_(True)
    #         model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    

    # model_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # logger.info(f"trainable model_params: {model_params}")

    # trainer = Trainer(
    #     model=model,
    #     args=training_args,
    #     train_dataset=train_dataset,
    #     eval_dataset=val_dataset,
    #     tokenizer=tokenizer,
    #     data_collator=DataCollatorForLanguageModeling(
    #         tokenizer=tokenizer, mlm=False
    #     )
    # )

    # # Training
    # train_result = trainer.train()
    # trainer.save_model()  # Saves the tokenizer too for easy upload

    # metrics = train_result.metrics
    # metrics["train_samples"] = len(train_dataset)

    # trainer.log_metrics("train", metrics)
    # trainer.save_metrics("train", metrics)
    # trainer.save_state()

    # # remove the full model in the end to save space, only adapter is needed
    # if isinstance(model, PeftModel):
    #     pytorch_model_path = os.path.join(
    #         training_args.output_dir, "pytorch_model_fsdp.bin")
    #     os.remove(pytorch_model_path) if os.path.exists(
    #         pytorch_model_path) else None

    # if model_args.model_name_abbr not in MODEL_CONFIGS:
    #     raise ValueError(f"Model abbreviation {model_args.model_name_abbr} not supported. "
    #                     f"Supported models: {list(MODEL_CONFIGS.keys())}")

if __name__ == "__main__":
    main()