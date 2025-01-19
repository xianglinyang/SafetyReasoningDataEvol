# SafetyReasoningDataEvol

## Useful links
- https://github.com/andyrdt/refusal_direction/tree/main/pipeline/model_utils
- https://github.com/centerforaisafety/HarmBench/blob/main/baselines/model_utils.py
- https://github.com/GraySwanAI/circuit-breakers/tree/main
- https://github.com/allenai/open-instruct/tree/main

# Run Instructions
1. evol the question+ download the dataset
```bash
bash script/evol/data_evol.sh
```
2. train model
```bash
bash script/train/lora_train.sh
```

# Download
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

# Evaluate instruction
1. download HarmBench repo
```bash
git clone https://github.com/centerforaisafety/HarmBench.git
```
Read evaluation pipeline in HarmBench repo
2. modify the configs
- models.yaml
- run_pipeline.yaml
3. run the pipeline
```bash
python scripts/run_pipeline.py
```
