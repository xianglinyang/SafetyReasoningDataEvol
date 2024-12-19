'''Model-specific templates'''

# Llama 2 chat templates are based on
# - https://github.com/centerforaisafety/HarmBench/blob/main/baselines/model_utils.py
LLAMA2_DEFAULT_SYSTEM_PROMPT = """You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature. If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."""


# Llama 3 chat templates are based on
# - https://llama.meta.com/docs/model-cards-and-prompt-formats/meta-llama-3/
# <|begin_of_text|> is automatically added by the tokenizer
LLAMA3_BEGIN_OF_TEXT = "<|begin_of_text|>"
LLAMA3_END_OF_TEXT = "<|eot_id|>"
LLAMA3_USER_TAG="<|start_header_id|>user<|end_header_id|>\n\n"
LLAMA3_ASSISTANT_TAG="<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
LLAMA3_SYSTEM = "<|start_header_id|>system<|end_header_id|>{}<|eot_id|>".format(LLAMA2_DEFAULT_SYSTEM_PROMPT)
LLAMA3_SEP_TOKEN = ""

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

def get_system_prompt(model_name_or_path):
    if "llama" in model_name_or_path.lower():
        return LLAMA2_DEFAULT_SYSTEM_PROMPT
    elif "mistral" in model_name_or_path.lower():
        return MISTRAL_DEFAULT_SYSTEM_PROMPT
    else:
        raise ValueError(f"Unsupported model: {model_name_or_path}")