#!/bin/bash


run_id=$RANDOM

python -m src.evol.question_mutation_main --dataset_name circuit_breakers_train --llm_name openai/gpt-4o-mini --num_variants 18 --alpha 0.5 --demo_selected_strategy diverse --run_id $run_id
