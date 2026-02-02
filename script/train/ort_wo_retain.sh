export CUDA_VISIBLE_DEVICES=1

OUTPUTS_PATH="/data2/xianglin/RobustSCoT"
mkdir -p ${OUTPUTS_PATH}/ort_outputs

MODEL_NAME="Meta-Llama-3-8B"
MODEL_NAME_OR_PATH="meta-llama/Meta-Llama-3-8B-Instruct"
DATASET_NAME="R2D-R1"
# --multi_gpu \
# --num_processes 2 \
accelerate launch \
    --num_processes 1 \
    --mixed_precision bf16 \
    src/orthogonal/main_wo_retain.py \
    --model_name_or_path ${MODEL_NAME_OR_PATH} \
    --dataset_name ${DATASET_NAME} \
    --use_lora \
    --output_dir ${OUTPUTS_PATH}/ort_outputs/${MODEL_NAME}-${DATASET_NAME}-ort-wo-retain \
    --bf16 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --learning_rate 5e-5 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.02 \
    --report_to none \
    --gradient_checkpointing \
    --ddp_find_unused_parameters False \
    --logging_steps 10 \
    --r 32 \
    --lora_alpha 32 \
    --lora_dropout 0.05 \
    --modules_to_save lm_head \
    --target_modules q_proj k_proj v_proj up_proj down_proj \
    --rho 0.01 \
    --adaptive False \
    --orth_sam_max_grad_norm 1.0 \
    --lam_u 5.0 \
    --one_sided 1 \
    --proj_scale 0.0 \
