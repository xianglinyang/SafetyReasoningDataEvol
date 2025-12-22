#!/bin/bash


run_id=$RANDOM
python -m src.evol.assemble --dataset_name STAIR-SFT --llm_name openai/gpt-4o-mini --num 18 --demo_selected_strategy diverse --run_id $run_id

python -m src.evol.assemble --dataset_name R2D-R1 --llm_name openai/gpt-4o-mini --num 18 --demo_selected_strategy diverse --run_id $run_id
