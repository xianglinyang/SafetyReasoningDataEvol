#!/bin/bash
# export CUDA_VISIBLE_DEVICES=1
per_gpu_jobs_num=1
gpu_num=2
jobs_num=$((per_gpu_jobs_num*gpu_num))


model_name_or_path_list=(
    # "/data2/xianglin/SCoT/outputs/circuitbreaker/llama3-8b_20250329-001230"
    # "/data2/xianglin/SCoT/outputs/circuitbreaker/mistral-7b_20250328-184812/checkpoint-297"
    
    # "meta-llama/Llama-3.1-8B-Instruct"
    "mistralai/Mistral-7B-Instruct-v0.2"

    # "GraySwanAI/Llama-3-8B-Instruct-RR"
    # "GraySwanAI/Mistral-7B-Instruct-RR"

    # "cais/HarmBench-Mistral-7b-val-cls"

    # Albation variants
    # "outputs/circuitbreaker/llama3-8b_ablation_variants/checkpoint-312"
    # "outputs/circuitbreaker/mistral-7b_ablation_variants/checkpoint-312"

    # Ablation retain datasets
    # "outputs/circuitbreaker/llama3-8b_ablate_retain/checkpoint-624"
    # "outputs/circuitbreaker/mistral-7b_ablation_retain/checkpoint-624"

    # Ablation CoT
    # "/data2/xianglin/SCoT/outputs/circuitbreaker/llama3-8b-ablation_CoT"
    # "/data2/xianglin/SCoT/outputs/circuitbreaker/mistral-7b-ablation_CoT"

    # num epochs
    # "/data2/xianglin/SCoT/outputs/circuitbreaker/mistral-7b_20250328-184812/checkpoint-148"
    # "/data2/xianglin/SCoT/outputs/circuitbreaker/mistral-7b_20250328-184812/checkpoint-297"
    # "/data2/xianglin/SCoT/outputs/circuitbreaker/mistral-7b_20250328-184812/checkpoint-444"
)
dataset_name_list=(
    # "sorrybench"
    "jailbreakbench"
    "advbench"
    # "harmbench"
    # "harmbench_attack"
)
attack_name_list=(
    "none"
    # "refusal_suppression"
    # "prefix_injection"
    # "base64"
    # "base64_input_only"
    # "style_injection_short"
    # "style_injection_json"
    # "distractors"
    # "disemvowel"
    # "leetspeak"
    # "poems"
)

# splits=(
    # "logical_appeal"
    # "uncommon_dialects"
    # "role_play"
    # "expert_endorsement"
    # "slang"
    # "evidence-based_persuasion"

    # "ascii"
    # "atbash"
    # "caesar"
    # "morse"

    # "authority_endorsement"
    # "misspellings"
    # "misrepresentation"
    # "technical_terms"

    # "translate-fr"
    # "translate-mr"
    # "translate-zh-cn"
    # "translate-ml"
    "translate-ta"
# )

save_dir_list=(
    # "/home/xianglin/git_space/HarmBench/test_cases/ablation_cot/AutoDAN/test_cases.json"
    # "/home/xianglin/git_space/HarmBench/test_cases/ablation_cot/GCG/test_cases.json"
    # "/home/xianglin/git_space/HarmBench/test_cases/ablation_cot/PAIR/test_cases.json"
    None
)

splits=("train")
prompt_cot_list=(1 2 3)
# prompt_cot_list=(0)
header="python -m src.evaluate.evaluate_harmful"
base_arguments="\
--device cuda \
--torch_type bf16 \
--eval_num 300"

# Counter to distribute commands across GPUs
counter=0

for save_dir in ${save_dir_list[@]}; do
    for split in ${splits[@]}; do
        for model_name_or_path in ${model_name_or_path_list[@]}; do
            for prompt_cot in ${prompt_cot_list[@]}; do
            for dataset_name in ${dataset_name_list[@]}; do
                for attack_name in ${attack_name_list[@]}; do
                    # Calculate GPU ID (0 or 1) and slot (0 or 1) for each command
                    gpu_id=$((counter % gpu_num))
                    # gpu_id=0
                    
                        # generate a random job id for each command
                        run_id=$RANDOM
                        
                        # Construct and run the command in background
                        CUDA_VISIBLE_DEVICES=$gpu_id $header $base_arguments \
                            --split ${split} \
                            --model_name_or_path ${model_name_or_path} \
                            --dataset_name ${dataset_name} \
                            --attack_name ${attack_name} \
                            --prompt_cot ${prompt_cot} \
                            --save_dir ${save_dir} \
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
    done
done

# Wait for any remaining background jobs
wait

# # --- Generate commands for GNU Parallel ---
# declare -a commands
# counter=0
# for split in "${splits[@]}"; do
#     for model_name_or_path in "${model_name_or_path_list[@]}"; do
#         for dataset_name in "${dataset_name_list[@]}"; do
#             for attack_name in "${attack_name_list[@]}"; do
#                 gpu_id=$((counter % gpu_num))
#                 run_id=$RANDOM

#                 # Build the command string, ensuring proper quoting
#                 cmd="CUDA_VISIBLE_DEVICES=$gpu_id $header $base_arguments \
#                     --split ${split} \
#                     --model_name_or_path ${model_name_or_path} \
#                     --dataset_name ${dataset_name} \
#                     --attack_name ${attack_name} \
#                     --prompt_cot ${prompt_cot} \
#                     --run_id $run_id"

#                 commands+=("$cmd") # Add command to an array
#                 counter=$((counter + 1))
#             done
#         done
#     done
# done

# # --- Execute using GNU Parallel ---
# echo "Running ${#commands[@]} commands with max $jobs_num parallel jobs..."

# printf '%s\n' "${commands[@]}" | parallel -j $jobs_num --eta --progress --halt now,fail=1
