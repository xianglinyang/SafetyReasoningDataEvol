#!/bin/bash

export WANDB_MODE=offline
export MASTER_PORT=$((29000 + RANDOM % 1000))
export CUBLAS_WORKSPACE_CONFIG=:16:8

# Get the directory of this script and find project root
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "${SCRIPT_DIR}/../.." && pwd )"

# Set PYTHONPATH to include the project root
export PYTHONPATH="${PYTHONPATH}:${PROJECT_ROOT}"

# Change to project root directory
cd ${PROJECT_ROOT}

### Llama-3-8B Config ###
model_name_or_path=meta-llama/Meta-Llama-3-8B-Instruct
model_name="Meta-Llama-3-8B-Instruct"

lorra_alpha=10
layers="10,20"
transform_layers="-1"

output_dir="/data2/xianglin/RobustSCoT/rr_outputs/${model_name}"

echo "model_name_or_path=$model_name_or_path"
echo "output_dir=$output_dir"

accelerate launch --config_file src/Baselines/circuitbreaker/configs/accelerate_zero1.yaml \
    --main_process_port $MASTER_PORT \
    src/Baselines/circuitbreaker/main.py \
    --model_name_or_path $model_name_or_path \
    --target_layers $layers \
    --transform_layers $transform_layers \
    --lorra_alpha $lorra_alpha \
    --lora_r 16 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --output_dir  $output_dir \
    --overwrite_output_dir \
    --max_steps 150 \
    --bf16 True \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 32 \
    --gradient_accumulation_steps 1 \
    --use_refusal_retain \
    --do_eval \
    --eval_steps 1000  \
    --save_total_limit 0 \
    --learning_rate 1e-4 \
    --weight_decay 0. \
    --lr_scheduler_type "constant" \
    --logging_strategy "steps" \
    --logging_steps 10 \
    --tf32 True \
    --model_max_length 1024 \
    --q_lora False \
    --gradient_checkpointing True \
    --report_to none \
    --log_every 1
