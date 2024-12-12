#!/bin/bash
# export CUDA_VISIBLE_DEVICES
jobs_num=2
gpu_num=2

model_name_or_path_list=(
    # "meta-llama/Llama-2-7b-chat-hf"
    "meta-llama/Llama-3.1-8B-Instruct"
    "mistralai/Mistral-7B-Instruct-v0.1"
    "outputs/circuitbreaker/llama2-7b"
    "outputs/circuitbreaker/llama3-8b"
    "outputs/circuitbreaker/mistral-7b"
)
dataset_name_list=(
    # "sorrybench"
    "jailbreakbench"
    # "advbench"
)
attack_name_list=(
    "none"
    "refusal_suppression"
    # "prefix_injection"
)


header="python -m src.evaluate.evaluate_harmful"
base_arguments="\
--split train \
--device cuda \
--torch_type bf16 \
--eval_num 200"

# Counter to distribute commands across GPUs
counter=0

for model_name_or_path in ${model_name_or_path_list[@]}; do
    for dataset_name in ${dataset_name_list[@]}; do
        for attack_name in ${attack_name_list[@]}; do
            # Calculate GPU ID (0 or 1) and slot (0 or 1) for each command
            gpu_id=$((counter % gpu_num))
            
            # generate a random job id for each command
            run_id=$RANDOM
            
            # Construct and run the command in background
            CUDA_VISIBLE_DEVICES=$gpu_id $header $base_arguments \
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

# Wait for any remaining background jobs
wait
