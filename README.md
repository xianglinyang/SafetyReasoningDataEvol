# SafetyReasoningDataEvol

## Useful links
- https://github.com/andyrdt/refusal_direction/tree/main/pipeline/model_utils
- https://github.com/centerforaisafety/HarmBench/blob/main/baselines/model_utils.py
- https://github.com/GraySwanAI/circuit-breakers/tree/main
- https://github.com/allenai/open-instruct/tree/main

# Run Instructions
Step 1: Download the dataset and evolve the questions  
** No need to run this step as we have already downloaded the dataset and processed it **
```bash
python -m sec.evol.assemble
```

Step 2: Train model
```bash
bash script/train/lora_train.sh
```

Step 3: Evaluate the model
```bash
bash script/evaluate/prompt_attack.sh   # for prompt attack
bash script/evaluate/evol_reasoning.sh  # for reasoning
```

(Optional) Step 4: Evaluate the model on HarmBench
- Download HarmBench repo
```bash
git clone https://github.com/centerforaisafety/HarmBench.git
```
- modify the configs
- run the test case generation
```bash
# Example
cd HarmBench
python generate_test_cases.py --method_name GCG --experiment_name evol_reasoning --save_dir ./test_cases
```

# Requirements
1. install requirements.txt
```bash
pip install -r requirements.txt
```
2. login huggingface
```bash
huggingface-cli login
```
3. export OPENAI_API_KEY
```bash
export OPENAI_API_KEY="xxx"
```
