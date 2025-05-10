#!/bin/bash
# '''Adapted from https://github.com/princeton-nlp/LESS/tree/main/less/scripts'''

# ---------os env---------
# Remove the CUDA_VISIBLE_DEVICES line since we want to use multiple GPUs
# export CUDA_VISIBLE_DEVICES=0

# ---------base arguments---------
ID=$RANDOM
header="torchrun --nproc_per_node=2 --nnodes=1 --rdzv-id=$ID --rdzv_backend=c10d -m src.train.sft"

base_arguments="\
--max_seq_length 1024 \
--lora True \
--lora_r 64 \
--lora_alpha 128 \
--lora_dropout 0.1 \
--lora_target_modules q_proj k_proj v_proj o_proj \
--use_fast_tokenizer True \
--learning_rate 2e-05 \
--gradient_accumulation_steps 16 \
--optim adamw_torch \
--num_train_epochs 2 \
--torch_dtype bfloat16 \
--bf16 True \
--tf32 False \
--fp16 False \
--overwrite_output_dir True \
--do_train True \
--do_eval True \
--evaluation_strategy steps \
--eval_steps 500 \
--save_strategy epoch \
--metric_for_best_model loss \
--greater_is_better False \
--ddp_find_unused_parameters False \
--warmup_ratio 0.03 \
--weight_decay 0.0 \
--logging_steps 1 \
--remove_unused_columns False
--include_variants True
--include_reasoning 0
"

# ---------dataset arguments---------
dataset_names=("circuitbreaker")
model_name_or_paths=("meta-llama/Llama-3.1-8B-Instruct" "mistralai/Mistral-7B-Instruct-v0.2")
model_name_abbrs=("llama3-8b" "mistral-7b")
per_device_train_batch_size=(1 1)

out_dir="/data2/xianglin/SCoT/outputs"

# Loop through each dataset
for dataset_name in ${dataset_names[@]}; do
    # loop through each model
    for i in "${!model_name_or_paths[@]}"; do

        # output dir format with model name+date+time
        output_dir="${out_dir}/${dataset_name}/${model_name_abbrs[$i]}_$(date +%Y%m%d-%H%M%S)"
        if [[ ! -d $output_dir ]]; then
            mkdir -p $output_dir
        fi

        # for the same command, the log file name is the same
        run_id=$RANDOM

        full_command="$header $base_arguments \
        --dataset_name $dataset_name \
        --output_dir $output_dir \
        --model_name_or_path ${model_name_or_paths[$i]} \
        --run_id $run_id \
        --per_device_train_batch_size ${per_device_train_batch_size[$i]} \
        --per_device_eval_batch_size ${per_device_train_batch_size[$i]}"

        echo "$full_command"
        eval $full_command
    done
done