# SafetyReasoningDataEvol

## Useful links
- https://github.com/andyrdt/refusal_direction/tree/main/pipeline/model_utils
- https://github.com/centerforaisafety/HarmBench/blob/main/baselines/model_utils.py
- https://github.com/GraySwanAI/circuit-breakers/tree/main
- https://github.com/allenai/open-instruct/tree/main

# Run Instructions
Step 1: Generate Demonstration for evolution (optional)
1. Generate with ```src.evol.diverse_demo.ipynb```
2. Copy paste the demo to ```src.evol.question_evol_prompt.py```

Step 2: Implement a new demonstration strategy in ```question_evol``` and ```assemble``` (Optional)

Step 3: Download the dataset, evolve the questions, and get completions of dolly answers
** No need to run this step as we have already downloaded the dataset and processed it **
```bash
python -m src.evol.assemble
```

Step 4: Filter non refusal answer

Step 5: Train model
```bash
bash script/train/lora_train.sh
```

Step 6: Evaluate the model
```bash
bash script/evaluate/prompt_attack.sh   # for prompt attack
bash script/evaluate/evol_reasoning.sh  # for reasoning
```

(Optional) Step 7: Evaluate the model on HarmBench
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
python merge_test_cases.py --method_name GCG --save_dir xxx
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


