#!/bin/bash
# '''Adapted from https://github.com/princeton-nlp/LESS/tree/main/less/scripts'''

# ---------os env---------
# Remove the CUDA_VISIBLE_DEVICES line since we want to use multiple GPUs
# export CUDA_VISIBLE_DEVICES=1

# ---------base arguments---------
ID=$RANDOM
header="torchrun --nproc_per_node=1 --nnodes=1 --rdzv-id=$ID --rdzv_backend=c10d -m src.train.sft"

base_arguments="\
--ratio 0.5 \
--max_seq_length 1024 \
--lora True \
--lora_r 64 \
--lora_alpha 256 \
--lora_dropout 0.1 \
--lora_target_modules q_proj k_proj v_proj o_proj \
--use_fast_tokenizer True \
--learning_rate 2e-05 \
--per_device_train_batch_size 2 \
--per_device_eval_batch_size 2 \
--gradient_accumulation_steps 16 \
--optim adamw_torch \
--num_train_epochs 1 \
--torch_dtype bfloat16 \
--bf16 True \
--tf32 False \
--fp16 False \
--overwrite_output_dir True \
--do_train True \
--do_eval True \
--evaluation_strategy steps \
--eval_steps 100 \
--save_strategy no \
--metric_for_best_model loss \
--greater_is_better False \
--ddp_find_unused_parameters False \
--warmup_ratio 0.03 \
--weight_decay 0.0 \
--logging_steps 1 \
--remove_unused_columns False
"

# ---------dataset arguments---------
dataset_names=("circuitbreaker")
model_name_or_path="meta-llama/Llama-3.1-8B-Instruct"
model_name_abbr="llama3-8b"

out_dir="outputs"

# Loop through each dataset
for dataset_name in ${dataset_names[@]}; do
    # loop through each model
    # output dir format with model name+date+time
    output_dir="${out_dir}/${dataset_name}/${model_name_abbr}_$(date +%Y%m%d-%H%M%S)"
    if [[ ! -d $output_dir ]]; then
        mkdir -p $output_dir
    fi

    # for the same command, the log file name is the same
    run_id=$RANDOM

    full_command="$header $base_arguments \
    --dataset_name $dataset_name \
    --output_dir $output_dir \
    --model_name_or_path ${model_name_or_path} \
    --run_id $run_id"

    echo "$full_command"
    eval $full_command

done