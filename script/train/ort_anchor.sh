export CUDA_VISIBLE_DEVICES=1

# Get the directory of this script and find project root
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "${SCRIPT_DIR}/../.." && pwd )"

# Set PYTHONPATH to include the project root
export PYTHONPATH="${PYTHONPATH}:${PROJECT_ROOT}"

# Change to project root directory
cd ${PROJECT_ROOT}

OUTPUTS_PATH="/data2/xianglin/RobustSCoT"

MODEL_NAME="Meta-Llama-3-8B"
MODEL_NAME_OR_PATH="meta-llama/Meta-Llama-3-8B-Instruct"
DATASET_NAME="R2D-R1"

accelerate launch \
    --num_processes 1 \
    --mixed_precision bf16 \
    src/orthogonal/main_anchor_avg.py \
    --model_name_or_path ${MODEL_NAME_OR_PATH} \
    --dataset_name ${DATASET_NAME} \
    --use_lora \
    --output_dir ${OUTPUTS_PATH}/ort_outputs/${MODEL_NAME}-ort-anchor-avg \
    --bf16 \
    --num_train_epochs 3 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --learning_rate 5e-5 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.05 \
    --report_to none \
    --gradient_checkpointing \
    --ddp_find_unused_parameters False \
    --logging_steps 10 \
    --r 64 \
    --lora_alpha 64 \
    --lora_dropout 0.05 \
    --modules_to_save embed_tokens lm_head \
    --target_modules q_proj k_proj v_proj up_proj down_proj \
    --rho 0.05 \
    --adaptive False \
    --orth_sam_max_grad_norm 1.0 \
    --lam_u 5 \
    --proj_scale 1.0 \
