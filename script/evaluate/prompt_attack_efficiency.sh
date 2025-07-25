#!/bin/bash
# export CUDA_VISIBLE_DEVICES=1
per_gpu_jobs_num=1
gpu_num=1
jobs_num=$((per_gpu_jobs_num*gpu_num))
available_gpu_ids=(4)

model_name_or_path_list=(
    # "/mnt/hdd1/ljiahao/xianglin/SCoT/circuitbreaker/mistral-7b-v2-alpha64/checkpoint-435"
    # "outputs/circuitbreaker/llama3-8b_20250121-122232/checkpoint-625"
    # "outputs/circuitbreaker/llama3-8b_20250121-122232/checkpoint-1248"
    # "outputs/circuitbreaker/llama3-8b_20250121-122232/"
    # "outputs/circuitbreaker/mistral-7b_20250122-011012/checkpoint-313"
    # "outputs/circuitbreaker/mistral-7b_20250122-011012/checkpoint-624"
    # "outputs/circuitbreaker/mistral-7b_20250122-011012/"
    # "meta-llama/Llama-3.1-8B-Instruct"
    # "mistralai/Mistral-7B-Instruct-v0.2"
    # "GraySwanAI/Llama-3-8B-Instruct-RR"
    # "GraySwanAI/Mistral-7B-Instruct-RR"
    # "cais/HarmBench-Mistral-7b-val-cls"

    # # Albation variants
    # "outputs/circuitbreaker/llama3-8b_ablation_variants/checkpoint-312"
    # "outputs/circuitbreaker/mistral-7b_ablation_variants/checkpoint-156"

    # # Ablation retain dataasets
    # "outputs/circuitbreaker/llama3-8b_ablate_retain/checkpoint-624"
    # "outputs/circuitbreaker/mistral-7b_ablation_retain/checkpoint-312"
    "/mnt/hdd1/ljiahao/xianglin/SCoT/circuitbreaker/llama3-8b_20250423-004757/checkpoint-900"
    
)
dataset_name_list=(
    # "sorrybench"
    # "jailbreakbench"
    "advbench"
)
attack_name_list=(
    "none"
    # "refusal_suppression"
    # "prefix_injection"
)

# splits=(
#     "logical_appeal"
#     "uncommon_dialects"
#     "role_play"
#     "expert_endorsement"
#     "slang"
#     "evidence-based_persuasion")
splits=("train")

header="python -m src.evaluate.evaluate_harmful_efficiency"
base_arguments="\
--device cuda \
--torch_type bf16 \
--eval_num 50"
# --split train \

# Counter to distribute commands across GPUs
counter=0

for split in ${splits[@]}; do
    for model_name_or_path in ${model_name_or_path_list[@]}; do
        for dataset_name in ${dataset_name_list[@]}; do
            for attack_name in ${attack_name_list[@]}; do
                # Calculate GPU ID (0 or 1) and slot (0 or 1) for each command
                gpu_id=${available_gpu_ids[$((counter % gpu_num))]}
                
                # generate a random job id for each command
                run_id=$RANDOM
                
                # Construct and run the command in background
                CUDA_VISIBLE_DEVICES=$gpu_id $header $base_arguments \
                    --split ${split} \
                    --model_name_or_path ${model_name_or_path} \
                    --dataset_name ${dataset_name} \
                    --attack_name ${attack_name} \
                    --run_id "$run_id" &
                
                # Increment counter
                counter=$((counter + 1))
                
                # If we've launched jobs_num jobs, wait for them to complete
                if [ $((counter % jobs_num)) -eq 0 ]; then
                    wait
                fi
            done
        done
    done
done

# Wait for any remaining background jobs
wait
