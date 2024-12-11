#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

model_name_or_path_list=(
    "meta-llama/Llama-2-7b-chat-hf"
    "meta-llama/Llama-3.1-8B-Instruct"
    "mistralai/Mistral-7B-Instruct-v0.1"
    "outputs/circuitbreaker/llama2-7b"
    "outputs/circuitbreaker/llama3-8b"
    "outputs/circuitbreaker/mistral-7b"
)
dataset_name_list=(
    "sorrybench"
    "jailbreakbench"
    "advbench"
)
attack_name_list=(
    "none"
    "refusal_suppression"
    "prefix_injection"
)


header="torchrun --nproc_per_node=1 -m src.evaluate.evaluate_harmful"
base_arguments="\
--split train \
--device cuda \
--torch_type bf16 \
--eval_num 500"

for model_name_or_path in ${model_name_or_path_list[@]}; do
    for dataset_name in ${dataset_name_list[@]}; do
        for attack_name in ${attack_name_list[@]}; do
            # Run the Python script with arguments
            full_command="$header $base_arguments \
                --model_name_or_path ${model_name_or_path} \
                --dataset_name ${dataset_name} \
                --attack_name ${attack_name}"
            echo $full_command
            eval $full_command
        done
    done
done
