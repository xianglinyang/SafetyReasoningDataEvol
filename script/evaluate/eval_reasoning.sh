#!/bin/bash

per_gpu_jobs_num=1
gpu_num=1  # number of GPUs
jobs_num=$((per_gpu_jobs_num*gpu_num))  # number of jobs to run in parallel, jobs_num = gpu_num*per_gpu_jobs_num
available_gpu_ids=(0)

model_name_or_path_list=(
    # "meta-llama/Llama-3.1-8B-Instruct"
    # "mistralai/Mistral-7B-Instruct-v0.2"

    # "GraySwanAI/Llama-3-8B-Instruct-RR"
    # "GraySwanAI/Mistral-7B-Instruct-RR"

    # "cais/zephyr_7b_r2d2"
    # "meta-llama/Meta-Llama-3-8B-Instruct"

    # "meta-llama/Llama-2-13b-chat-hf"
    # "/mnt/hdd1/ljiahao/xianglin/SCoT/outputs/circuitbreaker/llama2-13b_20250727-095351/checkpoint-97"
    # "/mnt/hdd1/ljiahao/xianglin/SCoT/outputs/circuitbreaker/llama2-13b_20250727-095351/checkpoint-194"
    # "/mnt/hdd1/ljiahao/xianglin/SCoT/outputs/circuitbreaker/llama2-13b_20250727-095351/checkpoint-291"

    # "/mnt/hdd1/ljiahao/xianglin/SCoT/circuitbreaker/llama3-8b-random-demo"
    # "/mnt/hdd1/ljiahao/xianglin/SCoT/circuitbreaker/llama3-8b-half-size"

    # "thu-ml/STAIR-Llama-3.1-8B-SFT"
    # "/data2/xianglin/RobustSCoT/circuitbreaker/llama3-8b_20251219-165422/checkpoint-933"
    # "/data2/xianglin/RobustSCoT/circuitbreaker/llama3-8b_20251220-174526/checkpoint-933"
    # "/data2/xianglin/RobustSCoT/circuitbreaker/llama3-8b_20251220-174526/checkpoint-1244"
    # "/data2/xianglin/RobustSCoT/sam_outputs/Meta-Llama-3-8B-sam"
    # "/data2/xianglin/RobustSCoT/ort_outputs/Meta-Llama-3-8B-ort-ablation"
    # "/data2/xianglin/RobustSCoT/ort_outputs/Meta-Llama-3-8B-ort-wo-retain-ablation"
    # "/data2/xianglin/RobustSCoT/r2d_outputs/Meta-Llama-3-8B-r2d"
    # "/data2/xianglin/RobustSCoT/ort_outputs/Llama-3.1-8B-ort"
    # "/data2/xianglin/RobustSCoT/circuitbreaker/Llama-3-8b_CB"
    # "/data2/xianglin/RobustSCoT/ort_outputs/Meta-Llama-3-8B-ort-anchor-avg"
    # "/data2/xianglin/RobustSCoT/ort_outputs/Meta-Llama-3-8B-R2D-R1-ort"
    # "/data2/xianglin/RobustSCoT/r2d_outputs/Meta-Llama-3-8B-r2d"
    # "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
    # "/data2/xianglin/RobustSCoT/ort_outputs/DeepSeek-R1-Distill-Qwen-7B-R2D-R1-ort/checkpoint-epoch-2"
    "/data2/xianglin/RobustSCoT/r2d_outputs/DeepSeek-R1-Distill-Qwen-7B-r2d"
)
dataset_name_list=(
    "mmlu"
    "gsm8k"
    "arc-c"
    "arc-e"
    # "boolq"
    # "MMLU-STEM"
    # "sciq"
    # "SimpleQA"
    # "adv_glue"
    # "aqua"
    # "strategyqa"
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

