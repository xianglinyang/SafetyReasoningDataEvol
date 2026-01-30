#!/bin/bash

MODEL_NAME_OR_PATH="meta-llama/Meta-Llama-3-8B-Instruct"
NICKNAME="llama3-8b"
DATASET_NAME="circuitbreaker_diverse"
OUTPUT_PATH="/data2/xianglin/RobustSCoT/scot_outputs/${DATASET_NAME}/${NICKNAME}/dataset"
K_ANS_TOKENS="64"
MAX_SAMPLES="${MAX_SAMPLES:-}"  # Empty means use all samples
GPU_MEMORY_UTILIZATION=0.9
MAX_MODEL_LEN="1024"
TENSOR_PARALLEL_SIZE="1"

MUTATION_LLM="openai/gpt-4.1"
MUTATION_NUM="3"
MUTATION_ALPHA="0.8"
MUTATION_DEMO_SELECTED_STRATEGY="diverse"

EPOCH="0"

CUDA_VISIBLE_DEVICES=1 python -m src.train.probe_and_select \
    --model_name_or_path ${MODEL_NAME_OR_PATH} \
    --dataset_name ${DATASET_NAME} \
    --output_dir ${OUTPUT_PATH} \
    --k_ans_tokens ${K_ANS_TOKENS} \
    --gpu_memory_utilization ${GPU_MEMORY_UTILIZATION} \
    --max_model_len ${MAX_MODEL_LEN} \
    --tensor_parallel_size ${TENSOR_PARALLEL_SIZE} \
    --mutation_llm ${MUTATION_LLM} \
    --mutation_num ${MUTATION_NUM} \
    --mutation_alpha ${MUTATION_ALPHA} \
    --mutation_demo_selected_strategy ${MUTATION_DEMO_SELECTED_STRATEGY} \
    --epoch ${EPOCH}

    