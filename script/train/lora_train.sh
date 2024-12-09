#!/bin/bash
# '''Adapted from https://github.com/princeton-nlp/LESS/tree/main/less/scripts'''
export CUDA_VISIBLE_DEVICES=1

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
--per_device_train_batch_size 2 \
--gradient_accumulation_steps 32 \
--optim adamw_torch \
--num_train_epochs 4 \
--torch_dtype bfloat16 \
"
echo "$header"  
eval $header