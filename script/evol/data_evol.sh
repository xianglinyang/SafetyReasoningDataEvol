#!/bin/bash

# Save run with the same run_id
# circuitbreaker have 4993 samples, therefore for each class, we need to generate 1250 variants

run_id=$RANDOM
python -m src.evol.assemble --run_id $run_id
