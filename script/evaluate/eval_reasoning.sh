#!/bin/bash

# export CUDA_VISIBLE_DEVICES=0
jobs_num=2  # number of jobs to run in parallel, jobs_num = gpu_num*per_gpu_jobs_num
gpu_num=2  # number of GPUs

model_name_or_path_list=(
    "meta-llama/Llama-2-7b-chat-hf"
    "meta-llama/Llama-3.1-8B-Instruct"
    "mistralai/Mistral-7B-Instruct-v0.1"
    "outputs/circuitbreaker/llama2-7b"
    "outputs/circuitbreaker/llama3-8b"
    "outputs/circuitbreaker/mistral-7b"
)
dataset_name_list=(
    "mmlu"
    "gsm8k"
)

header="python -m src.evaluate.evaluate_common_reasoning"
base_arguments="\
--torch_type bf16 \
--split test \
--eval_num 500 \
--device cuda"

# Counter to distribute commands across GPUs
counter=0

for model_name_or_path in ${model_name_or_path_list[@]}; do
    for dataset_name in ${dataset_name_list[@]}; do
        gpu_id=$((counter % gpu_num))
        run_id=$RANDOM

        CUDA_VISIBLE_DEVICES=$gpu_id $header $base_arguments \
            --model_name_or_path $model_name_or_path \
            --dataset_name $dataset_name \
            --run_id $run_id &

        counter=$((counter + 1))

        if [ $((counter % jobs_num)) -eq 0 ]; then
            wait
        fi
    done
done

