#!/bin/bash

model_name_or_path_list=("out" "meta-llama/Llama-2-7b-chat-hf")
dataset_name_list=("sorrybench" "jailbreakbench" "advbench")
attack_name_list=("none" "refusal_suppression" "prefix_injection")

for model_name_or_path in ${model_name_or_path_list[@]}; do
    for dataset_name in ${dataset_name_list[@]}; do
        for attack_name in ${attack_name_list[@]}; do
            # Run the Python script with arguments
            python  -m src.evaluate.evaluate_harmful \
                --model_name_or_path ${model_name_or_path} \
                --model_abbr "llama2" \
                --dataset_name ${dataset_name} \
                --split "train" \
                --device "cuda" \
                --attack_name ${attack_name}
                > log/evaluate_${model_name_or_path}_${dataset_name}_${attack_name}.log
        done
    done
done
