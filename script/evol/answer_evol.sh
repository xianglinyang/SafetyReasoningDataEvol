#!/bin/bash


python -m src.evol.answer_metadata --data_path /home/ljiahao/xianglin/git_space/SafetyReasoningDataEvol/data/processed/circuitbreaker_train.json --save_path /home/ljiahao/xianglin/git_space/SafetyReasoningDataEvol/data/processed/circuitbreaker_train_metadata.json --question_type harmful --model_name gpt-4.1-mini --question_key_name prompt
python -m src.evol.answer_metadata --data_path /home/ljiahao/xianglin/git_space/SafetyReasoningDataEvol/data/processed/circuitbreaker_val.json --save_path /home/ljiahao/xianglin/git_space/SafetyReasoningDataEvol/data/processed/circuitbreaker_val_metadata.json --question_type benign --model_name gpt-4.1-mini --question_key_name prompt

python -m src.evol.answer_metadata --data_path /home/ljiahao/xianglin/git_space/SafetyReasoningDataEvol/data/processed/dolly_train_intention.json --save_path /home/ljiahao/xianglin/git_space/SafetyReasoningDataEvol/data/processed/dolly_train_metadata.json --question_type benign --model_name gpt-4.1-mini --question_key_name evolved_question
python -m src.evol.answer_metadata --data_path /home/ljiahao/xianglin/git_space/SafetyReasoningDataEvol/data/processed/dolly_val_intention.json --save_path /home/ljiahao/xianglin/git_space/SafetyReasoningDataEvol/data/processed/dolly_val_metadata.json --question_type benign --model_name gpt-4.1-mini --question_key_name evolved_question












