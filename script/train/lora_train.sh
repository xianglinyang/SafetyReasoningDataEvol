#!/bin/bash
# '''Adapted from https://github.com/princeton-nlp/LESS/tree/main/less/scripts'''
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

header="\
torchrun --nproc_per_node=1 -m src.train.sft \
--dataset_name circuitbreaker \
--max_seq_length 1024 \
--model_name_or_path meta-llama/Llama-2-7b-chat-hf \
--lora True \
--lora_r 128 \
--lora_alpha 512 \
--lora_dropout 0.1 \
--lora_target_modules q_proj k_proj v_proj o_proj \
--model_name_abbr llama2 \
--use_fast_tokenizer True \
--output_dir out \
--learning_rate 2e-05 \
--per_device_train_batch_size 1 \
--gradient_accumulation_steps 64 \
--optim adamw_torch \
--num_train_epochs 4 \
--bf16 True \
--tf32 False \
--fp16 False \
--fsdp 'full_shard auto_wrap' \
--fsdp_config llama2_7b_finetune \
"

echo "$header"  
eval $header