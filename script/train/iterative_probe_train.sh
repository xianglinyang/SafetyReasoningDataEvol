#!/bin/bash

# Iterative Probe-Train Pipeline
# This script performs multiple rounds of probe->train iterations
# Each round uses the checkpoint from the previous round


# ========================================
# Configuration
# ========================================

# Project root
PROJECT_ROOT="/home/xianglin/git_space/SafetyReasoningDataEvol"
cd ${PROJECT_ROOT}

# Add src to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:${PROJECT_ROOT}"

# Training configuration
INITIAL_MODEL="meta-llama/Meta-Llama-3-8B-Instruct"
NICKNAME="llama3-8b"
DATASET_NAME="circuitbreaker_diverse"
NUM_ROUNDS="3"
OUTPUT_DIR="/data2/xianglin/RobustSCoT/scot_outputs/${NICKNAME}"
RUN_ID="${NICKNAME}_$(date +%Y%m%d_%H%M%S)"

# Training hyperparameters
PER_DEVICE_BATCH_SIZE=1
GRADIENT_ACCUMULATION_STEPS="4"
LEARNING_RATE="2e-5"
MAX_SEQ_LENGTH="1024"
SEED="42"

# Probe configuration
K_ANS_TOKENS="64"
MAX_SAMPLES="${MAX_SAMPLES:-}"  # Empty means use all samples
GPU_MEMORY_UTILIZATION="0.7"
MAX_MODEL_LEN="1024"
TENSOR_PARALLEL_SIZE="1"

# Strategy selection
TOP_RATIO="0.1"

echo "=========================================="
echo "Iterative Probe-Train Pipeline"
echo "=========================================="
echo "Initial model: ${INITIAL_MODEL}"
echo "Dataset: ${DATASET_NAME}"
echo "Number of rounds: ${NUM_ROUNDS}"
echo "Output directory: ${OUTPUT_DIR}"
echo "Run ID: ${RUN_ID}"
echo "=========================================="

# Create output directory
mkdir -p ${OUTPUT_DIR}
mkdir -p ${OUTPUT_DIR}/probe_results

# Save configuration
CONFIG_FILE="${OUTPUT_DIR}/config.txt"
cat > ${CONFIG_FILE} << EOF
Run ID: ${RUN_ID}
Start time: $(date)
Initial model: ${INITIAL_MODEL}
Dataset: ${DATASET_NAME}
Number of rounds: ${NUM_ROUNDS}
Output directory: ${OUTPUT_DIR}
Batch size: ${PER_DEVICE_BATCH_SIZE}
Gradient accumulation steps: ${GRADIENT_ACCUMULATION_STEPS}
Learning rate: ${LEARNING_RATE}
Max sequence length: ${MAX_SEQ_LENGTH}
Seed: ${SEED}
K answer tokens: ${K_ANS_TOKENS}
Top ratio: ${TOP_RATIO}
EOF

echo "Configuration saved to ${CONFIG_FILE}"

# ========================================
# Iterative Loop
# ========================================

CURRENT_MODEL="${INITIAL_MODEL}"

for ((round=0; round<${NUM_ROUNDS}; round++)); do
    echo ""
    echo "=========================================="
    echo "Round ${round}/${NUM_ROUNDS}"
    echo "=========================================="
    echo "Current model: ${CURRENT_MODEL}"
    
    # --------------------------------
    # Step 1: Probe
    # --------------------------------
    PROBE_OUTPUT="${OUTPUT_DIR}/probe_results/probe_round_${round}.json"
    
    echo ""
    echo ">>> Step 1: Running probe..."
    echo ">>> Output: ${PROBE_OUTPUT}"
    
    PROBE_CMD="python -m src.train.probe \
        --model_name_or_path ${CURRENT_MODEL} \
        --dataset_name ${DATASET_NAME} \
        --output_path ${PROBE_OUTPUT} \
        --k_ans_tokens ${K_ANS_TOKENS} \
        --gpu_memory_utilization ${GPU_MEMORY_UTILIZATION} \
        --max_model_len ${MAX_MODEL_LEN} \
        --tensor_parallel_size ${TENSOR_PARALLEL_SIZE}"
    
    if [ -n "${MAX_SAMPLES}" ]; then
        PROBE_CMD="${PROBE_CMD} --max_samples ${MAX_SAMPLES}"
    fi
    
    eval ${PROBE_CMD}
    
    if [ ! -f "${PROBE_OUTPUT}" ]; then
        echo "Error: Probe failed to generate output at ${PROBE_OUTPUT}"
        exit 1
    fi
    
    echo "✓ Probe completed: ${PROBE_OUTPUT}"
    
    # --------------------------------
    # Step 2: Train
    # --------------------------------
    echo ""
    echo ">>> Step 2: Running training..."
    echo ">>> Checkpoint will be saved to: ${OUTPUT_DIR}/checkpoint-epoch-${round}"
    
    # Build training command
    # Note: CURRENT_MODEL is either the initial model (round 0) or a merged model (round > 0)
    # Since merged models are complete models, we don't need base_model_path
    python -m src.train.run_train \
        --probe_json_path ${PROBE_OUTPUT} \
        --model_name_or_path ${CURRENT_MODEL} \
        --dataset_name ${DATASET_NAME} \
        --output_dir ${OUTPUT_DIR} \
        --run_id ${RUN_ID} \
        --epoch ${round} \
        --per_device_train_batch_size ${PER_DEVICE_BATCH_SIZE} \
        --gradient_accumulation_steps ${GRADIENT_ACCUMULATION_STEPS} \
        --learning_rate ${LEARNING_RATE} \
        --max_seq_length ${MAX_SEQ_LENGTH} \
        --seed ${SEED} \
        --save_strategy no \
        --save_steps inf \
        --save_total_limit none
    
    CHECKPOINT_DIR="${OUTPUT_DIR}/checkpoint-epoch-${round}"
    if [ ! -d "${CHECKPOINT_DIR}" ]; then
        echo "Error: Training failed to create checkpoint at ${CHECKPOINT_DIR}"
        exit 1
    fi
    
    echo "✓ Training completed: ${CHECKPOINT_DIR}"
    
    # --------------------------------
    # Step 3: Merge LoRA adapter (for probe in next round)
    # --------------------------------
    NEXT_ROUND=$((round + 1))
    if [ ${NEXT_ROUND} -lt ${NUM_ROUNDS} ]; then
        echo ""
        echo ">>> Step 3: Merging LoRA adapter for next round..."
        
        # Merge the adapter into the base model
        MERGED_DIR="${OUTPUT_DIR}/merged-epoch-${round}"
        
        python -m src.train.merge_checkpoint \
            --base_model_path "${INITIAL_MODEL}" \
            --checkpoint_path "${CHECKPOINT_DIR}" \
            --merged_output_path "${MERGED_DIR}" \
            --device cpu
        
        if [ ! -d "${MERGED_DIR}" ]; then
            echo "Error: Failed to merge adapter at ${MERGED_DIR}"
            exit 1
        fi
        
        echo "✓ Adapter merged: ${MERGED_DIR}"
        
        # --------------------------------
        # Step 4: Update model for next round
        # --------------------------------
        # Use merged model for next round's probe
        CURRENT_MODEL="${MERGED_DIR}"
        
        echo ""
        echo ">>> Next round will use merged model: ${CURRENT_MODEL}"
    fi
    
    echo ""
    echo "=========================================="
    echo "✓ Round ${round} completed!"
    echo "=========================================="
done

# ========================================
# Final Summary
# ========================================

echo ""
echo "=========================================="
echo "All rounds completed successfully!"
echo "=========================================="
echo "Run ID: ${RUN_ID}"
echo "Total rounds: ${NUM_ROUNDS}"
echo "Output directory: ${OUTPUT_DIR}"
echo ""
echo "Probe results:"
for ((round=0; round<${NUM_ROUNDS}; round++)); do
    echo "  Round ${round}: ${OUTPUT_DIR}/probe_results/probe_round_${round}.json"
done
echo ""
echo "Checkpoints:"
for ((round=0; round<${NUM_ROUNDS}; round++)); do
    echo "  Round ${round}: ${OUTPUT_DIR}/checkpoint-epoch-${round}"
done
echo ""
echo "End time: $(date)"
echo "=========================================="
