# SafetyReasoningDataEvol

## Useful links
- https://github.com/andyrdt/refusal_direction/tree/main/pipeline/model_utils
- https://github.com/centerforaisafety/HarmBench/blob/main/baselines/model_utils.py
- https://github.com/GraySwanAI/circuit-breakers/tree/main
- https://github.com/allenai/open-instruct/tree/main

# Run Instructions

1. prepare prompt templates variants
```bash
python src/evol/question_evol.py
python src/evol/answer_evol.py
```
2. download and assemble data
```bash
python src/evol/assemble.py
```
3. train model
```bash
python src/train.py
```
