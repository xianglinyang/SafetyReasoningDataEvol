import json
from tqdm import tqdm
from src.llm_zoo.api_base_models import OpenAILLM

llm = OpenAILLM(model_name="gpt-4o-mini", 
            temperature=0.7, 
            max_tokens=4000)

prompt = """You are an expert in summarizing the intent of a text. Please summarize the following text: {instruction}.

Please be concise (less than 15 words) and strictly follow the format of json:
```json
{{
    "summary": The intent of the text is to ...
}}
```
"""

def clean_json_format(summary: str) -> str:
    try:
        clean_summary = summary.split("```json")[-1].split("```")[0]
        clean_summary = json.loads(clean_summary)
        clean_summary = clean_summary['summary']
    except:
        clean_summary = summary
    return clean_summary

def get_intention(instruction: str) -> str:
    summary = llm.invoke(prompt.format(instruction=instruction))
    summary = clean_json_format(summary)
    return summary

def main():
    dolly_train_path = "/home/ljiahao/xianglin/git_space/SafetyReasoningDataEvol/data/processed/dolly_train.json"
    dolly_val_path = "/home/ljiahao/xianglin/git_space/SafetyReasoningDataEvol/data/processed/dolly_val.json"

    dolly_train_save_path = "/home/ljiahao/xianglin/git_space/SafetyReasoningDataEvol/data/processed/dolly_train_intention.json"
    dolly_val_save_path = "/home/ljiahao/xianglin/git_space/SafetyReasoningDataEvol/data/processed/dolly_val_intention.json"
    dolly_train = json.load(open(dolly_train_path))
    dolly_val = json.load(open(dolly_val_path))

    # process dolly_train
    for item in tqdm(dolly_train):
        instruction = item['evolved_question']
        summary = get_intention(instruction)
        item['intention'] = summary
    
    with open(dolly_train_save_path, 'w') as f:
        json.dump(dolly_train, f, indent=4)

    # process dolly_val
    for item in tqdm(dolly_val):
        instruction = item['evolved_question']
        summary = get_intention(instruction)
        item['intention'] = summary
    
    with open(dolly_val_save_path, 'w') as f:
        json.dump(dolly_val, f, indent=4)

if __name__ == "__main__":
    main()


