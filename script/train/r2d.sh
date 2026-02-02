export CUDA_VISIBLE_DEVICES=0

# Get the directory of this script and find project root
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "${SCRIPT_DIR}/../.." && pwd )"

# Set PYTHONPATH to include the project root
export PYTHONPATH="${PYTHONPATH}:${PROJECT_ROOT}"

# Change to project root directory
cd ${PROJECT_ROOT}

OUTPUTS_PATH="/data2/xianglin/RobustSCoT"
mkdir -p ${OUTPUTS_PATH}/r2d_outputs


MODEL_NAME="DeepSeek-R1-Distill-Qwen-7B"
MODEL_NAME_OR_PATH="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"

#--multi_gpu --num_processes 4
accelerate launch \
    --num_processes 1 \
    --mixed_precision bf16 \
    src/Baselines/R2D/main.py \
    --model_name_or_path ${MODEL_NAME_OR_PATH} \
    --use_lora \
    --output_dir ${OUTPUTS_PATH}/r2d_outputs/${MODEL_NAME}-r2d \
    --bf16 \
    --num_train_epochs 1 \
    --save_strategy steps \
    --save_steps 2000 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --learning_rate 2e-4 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.02 \
    --report_to none \
    --gradient_checkpointing \
    --ddp_find_unused_parameters False \
    --logging_steps 10 \
    --r 64 \
    --lora_alpha 64 \
    --modules_to_save embed_tokens lm_head \
    --target_modules q_proj k_proj v_proj up_proj down_proj \

