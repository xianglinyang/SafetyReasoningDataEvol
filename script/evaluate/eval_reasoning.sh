#!/bin/bash

per_gpu_jobs_num=1
gpu_num=1  # number of GPUs
jobs_num=$((per_gpu_jobs_num*gpu_num))  # number of jobs to run in parallel, jobs_num = gpu_num*per_gpu_jobs_num
available_gpu_ids=(4)

model_name_or_path_list=(
    # "meta-llama/Llama-3.1-8B-Instruct"
    # "mistralai/Mistral-7B-Instruct-v0.2"

    # "GraySwanAI/Llama-3-8B-Instruct-RR"
    # "GraySwanAI/Mistral-7B-Instruct-RR"

    # "/mnt/hdd1/ljiahao/xianglin/SCoT/circuitbreaker/mistral-7b-v2-CR-CoT-ablation/checkpoint-873"
    "/mnt/hdd1/ljiahao/xianglin/SCoT/circuitbreaker/mistral-7b-v2-alpha64/checkpoint-435"

    # "/mnt/hdd1/ljiahao/xianglin/SCoT/circuitbreaker/mistral-7b-v2_20250428-003555/checkpoint-303"
    # "/mnt/hdd1/ljiahao/xianglin/SCoT/circuitbreaker/mistral-7b-retain-ablation/checkpoint-450"
    # "/mnt/hdd1/ljiahao/xianglin/SCoT/circuitbreaker/mistral-7b-variants-ablation/checkpoint-150"
    # "/mnt/hdd1/ljiahao/xianglin/SCoT/circuitbreaker/mistral-7b-reasoning-ablation/checkpoint-225"

    # "/mnt/hdd1/ljiahao/xianglin/SCoT/circuitbreaker/llama3-8b_20250423-004757/checkpoint-900"
    # "/mnt/hdd1/ljiahao/xianglin/SCoT/circuitbreaker/llama3-8b-retain-ablation/checkpoint-450"
    # "/mnt/hdd1/ljiahao/xianglin/SCoT/circuitbreaker/llama3-8b-variants-ablation/checkpoint-225"
    # "/mnt/hdd1/ljiahao/xianglin/SCoT/circuitbreaker/llama3-8b-reasoning-ablation/checkpoint-450"
    # "meta-llama/Meta-Llama-3-8B-Instruct"

)
dataset_name_list=(
    "mmlu"
    "gsm8k"
)

header="python -m src.evaluate.evaluate_common_reasoning"
base_arguments="\
--torch_type bf16 \
--split test \
--eval_num 50 \
--device cuda"

# Counter to distribute commands across GPUs
counter=0

for model_name_or_path in ${model_name_or_path_list[@]}; do
    for dataset_name in ${dataset_name_list[@]}; do
        # gpu_id=$((counter % gpu_num))
        # gpu_id=7
        gpu_id=${available_gpu_ids[$((counter % gpu_num))]}
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

