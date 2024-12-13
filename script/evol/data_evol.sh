#!/bin/bash

# Save run with the same run_id
# circuitbreaker have 4993 samples, therefore for each class, we need to generate 1250 variants

run_id=$RANDOM
python -m src.evol.question_evol --model_name gpt-4-turbo --num_seed 250 --num_variants 1250 --run_id $run_id
python -m src.evol.answer_evol --num_variants_per_class 1250 --model_name gpt-4-turbo --run_id $run_id
python -m src.evol.assemble --run_id $run_id
