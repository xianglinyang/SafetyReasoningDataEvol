#!/bin/bash
per_gpu_jobs_num=1
gpu_num=1
jobs_num=$((per_gpu_jobs_num*gpu_num))
gpu_ids=(0)

# Configuration for GPU usage
# Set to "single" to use one GPU for both inference and eval
# Set to "dual" to use separate GPUs for inference and eval
gpu_setup="single"

model_name_or_path_list=(
    # "meta-llama/Llama-3.1-8B-Instruct"
    # "mistralai/Mistral-7B-Instruct-v0.2"

    # "GraySwanAI/Llama-3-8B-Instruct-RR"
    # "GraySwanAI/Mistral-7B-Instruct-RR"

    # "cais/zephyr_7b_r2d2"

    # "/mnt/hdd1/ljiahao/xianglin/SCoT/circuitbreaker/mistral-7b-v2-CR-CoT-ablation/checkpoint-873" # reasoning ablation
    # "/mnt/hdd1/ljiahao/xianglin/SCoT/circuitbreaker/mistral-7b-v2-alpha64/checkpoint-435"

    # "/mnt/hdd1/ljiahao/xianglin/SCoT/circuitbreaker/mistral-7b-v2_20250428-003555/checkpoint-303"
    # "/mnt/hdd1/ljiahao/xianglin/SCoT/circuitbreaker/mistral-7b-retain-ablation/checkpoint-450"
    "/mnt/hdd1/ljiahao/xianglin/SCoT/circuitbreaker/mistral-7b-variants-ablation/checkpoint-150"
    # "/mnt/hdd1/ljiahao/xianglin/SCoT/circuitbreaker/mistral-7b-reasoning-ablation/checkpoint-225"

    # "/mnt/hdd1/ljiahao/xianglin/SCoT/circuitbreaker/llama3-8b_20250423-004757/checkpoint-900"
    # "/mnt/hdd1/ljiahao/xianglin/SCoT/circuitbreaker/llama3-8b-retain-ablation/checkpoint-450"
    # "/mnt/hdd1/ljiahao/xianglin/SCoT/circuitbreaker/llama3-8b-variants-ablation/checkpoint-225"
    # "/mnt/hdd1/ljiahao/xianglin/SCoT/circuitbreaker/llama3-8b-reasoning-ablation/checkpoint-450"

)
dataset_name_list=(
    # "sorrybench"
    # "jailbreakbench"
    # "advbench"
    # "harmbench"
    "harmbench_attack"
    # "xstest"
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
#     "logical_appeal"
#     "uncommon_dialects"
#     "role_play"
#     "expert_endorsement"
#     "slang"
#     "evidence-based_persuasion"

#     "ascii"
#     "atbash"
#     "caesar"
#     "morse"

#     "authority_endorsement"
#     "misspellings"
#     "misrepresentation"
#     "technical_terms"

#     "translate-fr"
#     "translate-mr"
#     "translate-zh-cn"
#     "translate-ml"
#     "translate-ta"
# )

save_dir_list=(
    # # llama3 8b
    # "/home/ljiahao/xianglin/git_space/HarmBench/test_cases/GCG/llama3_8b_scot/test_cases.json"
    # "/home/ljiahao/xianglin/git_space/HarmBench/test_cases/AutoDAN/llama3_8b_scot/test_cases.json"
    # "/home/ljiahao/xianglin/git_space/HarmBench/test_cases/PAIR/llama3_8b_scot/test_cases.json"
    
    # mistral 7b
    # "/home/ljiahao/xianglin/git_space/HarmBench/test_cases/GCG/mistral_7b_v2_scot/test_cases.json"
    # "/home/ljiahao/xianglin/git_space/HarmBench/test_cases/AutoDAN/mistral_7b_v2_scot/test_cases.json"
    # "/home/ljiahao/xianglin/git_space/HarmBench/test_cases/PAIR/mistral_7b_v2_scot/test_cases.json"

    # # llama3 8b ablation retain
    # "/home/ljiahao/xianglin/git_space/HarmBench/test_cases/GCG/llama3_8b_scot_retain_ablation/test_cases.json"
    # "/home/ljiahao/xianglin/git_space/HarmBench/test_cases/AutoDAN/llama3_8b_scot_retain_ablation/test_cases.json"
    # "/home/ljiahao/xianglin/git_space/HarmBench/test_cases/PAIR/llama3_8b_scot_retain_ablation/test_cases.json"

    # # llama3 8b ablation variants
    # "/home/ljiahao/xianglin/git_space/HarmBench/test_cases/GCG/llama3_8b_scot_variants_ablation/test_cases.json"
    # "/home/ljiahao/xianglin/git_space/HarmBench/test_cases/AutoDAN/llama3_8b_scot_variants_ablation/test_cases.json"
    # "/home/ljiahao/xianglin/git_space/HarmBench/test_cases/PAIR/llama3_8b_scot_variants_ablation/test_cases.json"

    # # llama3 8b ablation reasoning
    # "/home/ljiahao/xianglin/git_space/HarmBench/test_cases/GCG/llama3_8b_scot_reasoning_ablation/test_cases.json"
    # "/home/ljiahao/xianglin/git_space/HarmBench/test_cases/AutoDAN/llama3_8b_scot_reasoning_ablation/test_cases.json"
    # "/home/ljiahao/xianglin/git_space/HarmBench/test_cases/PAIR/llama3_8b_scot_reasoning_ablation/test_cases.json"

    # # mistral 7b ablation reasoning
    # "/home/ljiahao/xianglin/git_space/HarmBench/test_cases/GCG/mistral_7b_v2_scot_reasoning_ablation/test_cases.json"
    # "/home/ljiahao/xianglin/git_space/HarmBench/test_cases/AutoDAN/mistral_7b_v2_scot_reasoning_ablation/test_cases.json"
    # "/home/ljiahao/xianglin/git_space/HarmBench/test_cases/PAIR/mistral_7b_v2_scot_reasoning_ablation/test_cases.json"

    # # mistral 7b ablation variants
    # "/home/ljiahao/xianglin/git_space/HarmBench/test_cases/GCG/mistral_7b_v2_scot_variants_ablation/test_cases.json"
    # "/home/ljiahao/xianglin/git_space/HarmBench/test_cases/AutoDAN/mistral_7b_v2_scot_variants_ablation/test_cases.json"
    # "/home/ljiahao/xianglin/git_space/HarmBench/test_cases/PAIR/mistral_7b_v2_scot_variants_ablation/test_cases.json"

    # # mistral 7b ablation retain
    # "/home/ljiahao/xianglin/git_space/HarmBench/test_cases/GCG/mistral_7b_v2_scot_retain_ablation/test_cases.json"
    # "/home/ljiahao/xianglin/git_space/HarmBench/test_cases/AutoDAN/mistral_7b_v2_scot_retain_ablation/test_cases.json"
    # "/home/ljiahao/xianglin/git_space/HarmBench/test_cases/PAIR/mistral_7b_v2_scot_retain_ablation/test_cases.json"
    # "/home/ljiahao/xianglin/git_space/HarmBench/test_cases/GCG/mistral_7b_v2_epoch4/test_cases.json"
    # "/home/ljiahao/xianglin/git_space/HarmBench/test_cases/GCG/mistral_7b_v2_epoch5/test_cases.json"

    # "/home/ljiahao/xianglin/git_space/HarmBench/test_cases/AutoDAN/mistral_7b_v2_scot_variants_ablation/test_cases.json"
    # "/home/ljiahao/xianglin/git_space/HarmBench/test_cases/PAIR/mistral_7b_v2_scot_variants_ablation/test_cases.json"

    # "/home/ljiahao/xianglin/git_space/HarmBench/test_cases/GCG/mistral_7b_v2_scot_new/test_cases.json"
    # "/home/ljiahao/xianglin/git_space/HarmBench/test_cases/AutoDAN/mistral_7b_v2_scot_CR_CoT_ablation/test_cases.json"
    # "/home/ljiahao/xianglin/git_space/HarmBench/test_cases/PAIR/mistral_7b_v2_scot_CR_CoT_ablation/test_cases.json"
    # "/home/ljiahao/xianglin/git_space/HarmBench/test_cases/GCG/mistral_7b_v2_scot_alpha64/test_cases.json"
    # "/home/ljiahao/xianglin/git_space/HarmBench/test_cases/AutoDAN/mistral_7b_v2_scot_alpha64/test_cases.json"
    # "/home/ljiahao/xianglin/git_space/HarmBench/test_cases/PAIR/mistral_7b_v2_scot_alpha64/test_cases.json"
    # "/home/ljiahao/xianglin/git_space/HarmBench/test_cases/PAP/llama3_8b_scot/test_cases.json"
    # "/home/ljiahao/xianglin/git_space/HarmBench/test_cases/PAP/llama3_8b_variants_ablation/test_cases.json"
    # "/home/ljiahao/xianglin/git_space/HarmBench/test_cases/PAP/mistral_7b_v2_scot_alpha64/test_cases.json"
    # "/home/ljiahao/xianglin/git_space/HarmBench/test_cases/PAP/mistral_7b_v2_scot_CR_CoT_ablation/test_cases.json"
    "/home/ljiahao/xianglin/git_space/HarmBench/test_cases/PAP/mistral_7b_v2_scot_variants_ablation/test_cases.json"
    # None
)

splits=("train")
# prompt_cot_list=(1 2 3)
prompt_cot_list=(0)
header="python -m src.evaluate.evaluate_harmful"
base_arguments="\
--torch_type bf16 \
--eval_num 50"

# Counter to distribute commands across GPUs
counter=0

for save_dir in ${save_dir_list[@]}; do
    for split in ${splits[@]}; do
        for model_name_or_path in ${model_name_or_path_list[@]}; do
            for prompt_cot in ${prompt_cot_list[@]}; do
            for dataset_name in ${dataset_name_list[@]}; do
                for attack_name in ${attack_name_list[@]}; do
                        # Calculate GPU IDs based on setup
                        if [ "$gpu_setup" = "single" ]; then
                            gpu_id=${gpu_ids[$((counter % gpu_num))]}
                            device_args="--device cuda:0 --eval_device cuda:0"
                            cuda_visible_devices=$gpu_id
                        else
                            # For dual setup, use two consecutive GPUs
                            first_gpu_idx=$((counter % (gpu_num/2) * 2))
                            gpu_id1=${gpu_ids[$first_gpu_idx]}
                            gpu_id2=${gpu_ids[$((first_gpu_idx + 1))]}
                            device_args="--device cuda:0 --eval_device cuda:1"
                            cuda_visible_devices="$gpu_id1,$gpu_id2"
                        fi
                        
                        # generate a random job id for each command
                        run_id=$RANDOM
                        
                        # Construct and run the command in background
                        CUDA_VISIBLE_DEVICES=$cuda_visible_devices $header $base_arguments \
                            $device_args \
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
