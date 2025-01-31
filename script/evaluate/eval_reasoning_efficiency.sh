#!/bin/bash

# export CUDA_VISIBLE_DEVICES=0

per_gpu_jobs_num=1
gpu_num=1  # number of GPUs
jobs_num=1  # number of jobs to run in parallel, jobs_num = gpu_num*per_gpu_jobs_num

model_name_or_path_list=(
    # "outputs/circuitbreaker/llama3-8b_20250121-122232/checkpoint-625"
    "outputs/circuitbreaker/llama3-8b_20250121-122232/checkpoint-1248"
    # "outputs/circuitbreaker/llama3-8b_20250121-122232/"
    # "outputs/circuitbreaker/mistral-7b_20250122-011012/checkpoint-313"
    "outputs/circuitbreaker/mistral-7b_20250122-011012/checkpoint-624"
    # "outputs/circuitbreaker/mistral-7b_20250122-011012/"
    "meta-llama/Llama-3.1-8B-Instruct"
    "mistralai/Mistral-7B-Instruct-v0.2"
    "GraySwanAI/Llama-3-8B-Instruct-RR"
    "GraySwanAI/Mistral-7B-Instruct-RR"
    "cais/HarmBench-Mistral-7b-val-cls"
    # "outputs/circuitbreaker/llama3-8b_ablate_retain/checkpoint-624"
    # "outputs/circuitbreaker/llama3-8b_ablation_variants/checkpoint-312"
    # "outputs/circuitbreaker/mistral-7b_ablation_retain/checkpoint-312"
    # "outputs/circuitbreaker/mistral-7b_ablation_variants/checkpoint-156"


)
dataset_name_list=(
    "mmlu"
    "gsm8k"
)

header="python -m src.evaluate.evaluate_common_reasoning_efficiency"
base_arguments="\
--torch_type bf16 \
--split test \
--eval_num 10 \
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

