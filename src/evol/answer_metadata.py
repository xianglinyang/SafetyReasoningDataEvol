'''
Enrich the answers with reasoning why it is harmful/not harmful and output or refusal with 4o-mini.
'''
import logging
from tqdm import tqdm

import argparse

from src.llm_zoo.api_base_models import OpenAILLM
from src.utils.common_utils import str2json, load_json_data, save_json_data
from src.logger.config import setup_logging
from src.evol.answer_evol_prompt import HARMFUL_QUESTION_USER_PROMPT, BENIGN_QUESTION_USER_PROMPT

logger = logging.getLogger(__name__)

def get_harmful_metadata(instruction: str, llm_client: OpenAILLM) -> str:
    prompt = HARMFUL_QUESTION_USER_PROMPT + instruction
    summary = llm_client.invoke(prompt)
    summary = str2json(summary)
    return summary

def get_benign_metadata(instruction: str, llm_client: OpenAILLM) -> str:
    prompt = BENIGN_QUESTION_USER_PROMPT + instruction
    summary = llm_client.invoke(prompt)
    summary = str2json(summary)
    return summary

def reformat_categories(categories):
    reformatted_categories = {}
    for d in categories:
        for k,v in d.items():
            reformatted_categories[k] = v
    return reformatted_categories

def clean_metadata(metadata):
    m = metadata[0]
    intent = m["Intent"].split("core intent of the question is to ")[-1]
    if intent.endswith("."):
        intent = intent[:-1]
    categories = reformat_categories(m.get("Categories", []))
    refusal = m.get("Refusal", "")
    return intent, categories, refusal

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str)
    parser.add_argument("--save_path", type=str)
    parser.add_argument("--question_type", type=str, default="benign", choices=["benign", "harmful"])
    parser.add_argument("--question_key_name", type=str)
    parser.add_argument("--model_name", type=str, default="gpt-4.1-mini")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max_tokens", type=int, default=4096)
    parser.add_argument("--run_id", type=str)
    args = parser.parse_args()

    setup_logging(task_name="answer_metadata", run_id=args.run_id)

    llm_client = OpenAILLM(model_name=args.model_name, 
            temperature=args.temperature, 
            max_tokens=args.max_tokens)
    
    data = load_json_data(args.data_path)
    
    # process dolly_train
    for item in tqdm(data):
        instruction = item[args.question_key_name]
        if args.question_type == "harmful":
            metadata = get_harmful_metadata(instruction, llm_client)
        else:
            metadata = get_benign_metadata(instruction, llm_client)
        item['metadata'] = metadata
    
    save_json_data(data, args.save_path)
if __name__ == "__main__":
    main()