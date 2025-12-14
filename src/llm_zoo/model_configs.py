'''Model-specific templates'''

import logging

logger = logging.getLogger(__name__)

# Llama 2 chat templates are based on
# - https://github.com/centerforaisafety/HarmBench/blob/main/baselines/model_utils.py
LLAMA2_DEFAULT_SYSTEM_PROMPT = """You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature. If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."""
LLAMA2_END_OF_TEXT = "<|end_of_text|>"
LLAMA2_USER_TAG="<|start_header_id|>user<|end_header_id|>\n\n"
LLAMA2_ASSISTANT_TAG="<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
LLAMA2_SYSTEM_TAG="<|start_header_id|>system<|end_header_id|>\n\n"
LLAMA2_SYSTEM = "<|start_header_id|>system<|end_header_id|>{}<|eot_id|>".format(LLAMA2_DEFAULT_SYSTEM_PROMPT)
LLAMA2_SEP_TOKEN = ""
LLAMA2_FORMATTED_PROMPT = "<s>[INST] {prompt} [/INST]"

# Llama 3 chat templates are based on
# - https://llama.meta.com/docs/model-cards-and-prompt-formats/meta-llama-3/
# <|begin_of_text|> is automatically added by the tokenizer
LLAMA3_BEGIN_OF_TEXT = "<|begin_of_text|>"
LLAMA3_END_OF_TEXT = "<|eot_id|>"
LLAMA3_USER_TAG="<|start_header_id|>user<|end_header_id|>\n\n"
LLAMA3_ASSISTANT_TAG="<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
LLAMA3_SYSTEM = "<|start_header_id|>system<|end_header_id|>{}<|eot_id|>".format(LLAMA2_DEFAULT_SYSTEM_PROMPT)
LLAMA3_SEP_TOKEN = ""
LLAMA3_FORMATTED_PROMPT = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nCutting Knowledge Date: December 2023\nToday Date: 26 Jul 2024\n\n<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|>"

# Mistral chat templates are based on
# - https://github.com/mistralai/mistralai/blob/main/mistralai/chat.py
MISTRAL_DEFAULT_SYSTEM_PROMPT = """You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature. If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."""
MISTRAL_BEGIN_OF_TEXT = "<s>"
MISTRAL_END_OF_TEXT = "</s>"
MISTRAL_SYSTEM_TAG = "[INST]"
MISTRAL_SYSTEM = "[INST] {}".format(MISTRAL_DEFAULT_SYSTEM_PROMPT)
MISTRAL_USER_TAG="[INST]"
MISTRAL_ASSISTANT_TAG="[/INST]"
MISTRAL_SEP_TOKEN = " "
MISTRAL_FORMATTED_PROMPT = "<s> [INST] {prompt} [/INST]"

QWEN_DEFAULT_SYSTEM_PROMPT = """You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature. If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."""
QWEN_BEGIN_OF_TEXT = "<|im_start|>"
QWEN_END_OF_TEXT = "<|im_end|>"
QWEN_SYSTEM_TAG = "<|im_start|>system\n"
QWEN_SYSTEM = "<|im_start|>system\n{}<|im_end|>\n".format(QWEN_DEFAULT_SYSTEM_PROMPT)
QWEN_USER_TAG="<|im_start|>user\n"
QWEN_ASSISTANT_TAG="<|im_start|>assistant\n"
QWEN_SEP_TOKEN = "<|im_end|>"
QWEN_PAD_TOKEN = "<|endoftext|>"
QWEN_FORMATTED_PROMPT = "<|im_start|>user\n{prompt}<|im_end|>\n"


def get_stop_tokens(model_name_or_path):
    model_name_lower = model_name_or_path.lower()
    if "llama-3" in model_name_lower or "llama3" in model_name_lower:
        return [LLAMA3_END_OF_TEXT]
    elif "mistral-7" in model_name_lower or "mistral" in model_name_lower or "zephyr_7b_r2d2" in model_name_lower:
        return [MISTRAL_END_OF_TEXT]
    elif "llama-2" in model_name_lower or "llama2" in model_name_lower:
        return [LLAMA2_END_OF_TEXT]
    elif "qwen" in model_name_lower:
        return [QWEN_END_OF_TEXT]
    else:
        raise ValueError(f"Not implemented for model: {model_name_or_path}")

def get_formatted_prompt(model_name_or_path, prompt):
    # TODO or we implement with the tokenizer?
    model_name_lower = model_name_or_path.lower()
    if "llama-3" in model_name_lower or "llama3" in model_name_lower:
        return LLAMA3_FORMATTED_PROMPT.format(prompt=prompt)
    elif "mistral-7" in model_name_lower or "mistral" in model_name_lower:
        return MISTRAL_FORMATTED_PROMPT.format(prompt=prompt)
    elif "zephyr_7b_r2d2" in model_name_lower:
        return MISTRAL_FORMATTED_PROMPT.format(prompt=prompt)
    elif "llama-2" in model_name_lower or "llama2" in model_name_lower:
        return LLAMA2_FORMATTED_PROMPT.format(prompt=prompt)
    elif "qwen" in model_name_lower:
        return QWEN_FORMATTED_PROMPT.format(prompt=prompt)
    else:
        raise ValueError(f"Not implemented for model: {model_name_or_path}")