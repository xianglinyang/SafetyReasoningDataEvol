#!/bin/bash
# '''Adapted from https://github.com/princeton-nlp/LESS/tree/main/less/scripts'''

# ---------os env---------
# Remove the CUDA_VISIBLE_DEVICES line since we want to use multiple GPUs
# export CUDA_VISIBLE_DEVICES=1

# ---------base arguments---------
ID=$RANDOM
header="CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --nnodes=1 --rdzv-id=$ID --rdzv_backend=c10d -m src.train.robust_sft"

base_arguments="\
--max_seq_length 2048 \
--lora True \
--lora_r 64 \
--lora_alpha 256 \
--lora_dropout 0.1 \
--lora_target_modules q_proj k_proj v_proj o_proj \
--use_fast_tokenizer True \
--learning_rate 1e-6 \
--per_device_train_batch_size 1 \
--per_device_eval_batch_size 1 \
--gradient_accumulation_steps 16 \
--optim adamw_torch \
--num_train_epochs 3 \
--torch_dtype bfloat16 \
--bf16 True \
--tf32 False \
--fp16 False \
--overwrite_output_dir True \
--do_train True \
--do_eval True \
--eval_steps inf \
--save_strategy no
--metric_for_best_model loss \
--greater_is_better False \
--ddp_find_unused_parameters False \
--warmup_ratio 0.03 \
--weight_decay 0.0 \
--logging_steps 1 \
--remove_unused_columns False
--demo_selected_strategy diverse
--mutation_num 5
--mutation_alpha 0.9
--mutation_llm openai/gpt-4.1
--mutation_top_ratio 0.3
--benign_lambda 1
--harmful_lambda 1
--dataset_log_dir /data2/xianglin/RobustSCoT/Datasets
"

# ---------dataset arguments---------
dataset_names=("STAIR-SFT_diverse")
model_name_or_path="meta-llama/Llama-3.1-8B-Instruct"
model_name_abbr="llama3-8b"

out_dir="/data2/xianglin/RobustSCoT"

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
    --model_nickname ${model_name_abbr} \
    --run_id $run_id"

    echo "$full_command"
    eval $full_command

done