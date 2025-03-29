#!/bin/bash
# export CUDA_VISIBLE_DEVICES=1
per_gpu_jobs_num=1
gpu_num=1
jobs_num=$((per_gpu_jobs_num*gpu_num))


model_name_or_path_list=(
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
    # "outputs/circuitbreaker/mistral-7b_ablation_variants/checkpoint-312"

    # Ablation retain dataasets
    # "outputs/circuitbreaker/llama3-8b_ablate_retain/checkpoint-624"
    # "outputs/circuitbreaker/mistral-7b_ablation_retain/checkpoint-624"
    "/data2/xianglin/SCoT/outputs/circuitbreaker/llama3-8b_20250329-001230"
)
dataset_name_list=(
    # "sorrybench"
    "jailbreakbench"
    # "advbench"
    # "llama3-8b-AutoDAN"
    # "mistral-7b-AutoDAN"
    # "llama3-8b-GCG"
    # "mistral-7b-GCG"
)
attack_name_list=(
    # "none"
    "refusal_suppression"
    # "prefix_injection"
)

# splits=(
#     "logical_appeal"
    # "uncommon_dialects"
    # "role_play"
#     "expert_endorsement"
#     "slang"
#     "evidence-based_persuasion"

#     "ascii"
#     "authority_endorsement"
#     "misspellings"
#     "translate-fr"
#     "translate-mr"
#     "translate-zh-cn"
#     "atbash"
#     "caesar"
#     "misrepresentation"
#     "morse"
#     "technical_terms"
#     "translate-ml"
#     "translate-ta"
# )

splits=("train")
prompt_cot=0

header="python -m src.evaluate.evaluate_harmful"
base_arguments="\
--device cuda \
--torch_type bf16 \
--eval_num 50"

# Counter to distribute commands across GPUs
counter=0

for split in ${splits[@]}; do
    for model_name_or_path in ${model_name_or_path_list[@]}; do
        for dataset_name in ${dataset_name_list[@]}; do
            for attack_name in ${attack_name_list[@]}; do
                # Calculate GPU ID (0 or 1) and slot (0 or 1) for each command
                # gpu_id=$((counter % gpu_num))
                gpu_id=1
                
                # generate a random job id for each command
                run_id=$RANDOM
                
                # Construct and run the command in background
                CUDA_VISIBLE_DEVICES=$gpu_id $header $base_arguments \
                    --split ${split} \
                    --model_name_or_path ${model_name_or_path} \
                    --dataset_name ${dataset_name} \
                    --attack_name ${attack_name} \
                    --prompt_cot ${prompt_cot} \
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
