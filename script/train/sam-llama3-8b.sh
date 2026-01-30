export CUDA_VISIBLE_DEVICES=0,1,2,3
export OUTPUTS_PATH=/mnt/hdd/jiahao/xianglin/RobustSCoT

# Get the directory of this script and find project root
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "${SCRIPT_DIR}/../.." && pwd )"

# Set PYTHONPATH to include the project root
export PYTHONPATH="${PYTHONPATH}:${PROJECT_ROOT}"

# Change to project root directory
cd ${PROJECT_ROOT}

mkdir -p ${OUTPUTS_PATH}/sam_outputs

MODEL_NAME="Meta-Llama-3-8B"
MODEL_NAME_OR_PATH="meta-llama/Meta-Llama-3-8B-Instruct"
# --multi_gpu \
# --num_processes 2 \
accelerate launch \
    --multi_gpu \
    --num_processes 4 \
    --mixed_precision bf16 \
    src/Baselines/SAM/main.py \
    --model_name_or_path ${MODEL_NAME_OR_PATH} \
    --dataset_name circuitbreaker \
    --use_lora \
    --output_dir ${OUTPUTS_PATH}/sam_outputs/${MODEL_NAME}-sam \
    --bf16 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --learning_rate 2e-4 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.02 \
    --report_to none \
    --gradient_checkpointing \
    --ddp_find_unused_parameters False \
    --logging_steps 10 \
    --dataloader_num_workers 4 \
    --dataloader_pin_memory \
    --tf32 True \
    --r 64 \
    --lora_alpha 64 \
    --lora_dropout 0.05 \
    --bias none \
    --modules_to_save embed_tokens lm_head \
    --target_modules q_proj k_proj v_proj up_proj down_proj \
    --rho 0.05 \
    --adaptive False \
    --max_grad_norm 1.0 \
