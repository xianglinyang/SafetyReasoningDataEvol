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
    "mmlu"
    "gsm8k"
)

header="python -m src.evaluate.evaluate_common_reasoning"
base_arguments="\
--torch_type bf16 \
--split test \
--eval_num 500 \
--device cuda"

for model_name_or_path in ${model_name_or_path_list[@]}; do
    for dataset_name in ${dataset_name_list[@]}; do
        full_command="$header $base_arguments \
        --model_name_or_path $model_name_or_path \
        --dataset_name $dataset_name"
        echo $full_command
        eval $full_command
    done
done

