'''Model-specific templates'''

# Llama 2 chat templates are based on
# - https://github.com/centerforaisafety/HarmBench/blob/main/baselines/model_utils.py
LLAMA2_DEFAULT_SYSTEM_PROMPT = """You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.
If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."""
LLAMA2_CHAT_TEMPLATE = "[INST] {instruction} [/INST] "
LLAMA2_CHAT_TEMPLATE_WITH_SYSTEM = "[INST] <<SYS>>\n{}\n<</SYS>>\n\n{instruction} [/INST] "
LLAMA2_USER_TAG = "[INST] "
LLAMA2_ASSISTANT_TAG = " [/INST]"
LLAMA2_SEP_TOKEN = " "
LLAMA2_SYSTEM = "<<SYS>>\n{}\n<</SYS>>\n\n".format(LLAMA2_DEFAULT_SYSTEM_PROMPT)

# Llama 3 chat templates are based on
# - https://llama.meta.com/docs/model-cards-and-prompt-formats/meta-llama-3/
# <|begin_of_text|> is automatically added by the tokenizer
LLAMA3_CHAT_TEMPLATE = """<|start_header_id|>user<|end_header_id|>{instruction}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""
LLAMA3_CHAT_TEMPLATE_WITH_SYSTEM = """<|start_header_id|>system<|end_header_id|>{}<|eot_id|><|start_header_id|>user<|end_header_id|>{instruction}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""
LLAMA3_USER_TAG="<|start_header_id|>user<|end_header_id|>\n\n"
LLAMA3_ASSISTANT_TAG="<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
LLAMA3_SYSTEM = "<|start_header_id|>system<|end_header_id|>{}<|eot_id|>".format(LLAMA2_DEFAULT_SYSTEM_PROMPT)
LLAMA3_SEP_TOKEN = ""

# TODO Fix Mistral
# # Mistral chat templates are based on
# # - https://github.com/mistralai/mistralai/blob/main/mistralai/chat.py
# MISTRAL_DEFAULT_SYSTEM_PROMPT = """You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.
# If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."""
# MISTRAL_CHAT_TEMPLATE = "{{ bos_token }}{% for message in messages %}{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}{% endif %}{% if message['role'] == 'user' %}{{ '[INST] ' + message['content'] + ' [/INST]' }}{% elif message['role'] == 'assistant' %}{{ ' ' + message['content'] + eos_token}}{% else %}{{ raise_exception('Only user and assistant roles are supported!') }}{% endif %}{% endfor %}"
# MISTRAL_SYSTEM_TAG = "<|start_header_id|>system<|end_header_id|>\n\n"
# MISTRAL_USER_TAG="[INST] "
# MISTRAL_ASSISTANT_TAG=" [/INST]"
# MISTRAL_SEP_TOKEN = " "

# TODO Add more models
MODEL_CONFIGS = {
    "llama2": {
        "user_tag": LLAMA2_USER_TAG,
        "assistant_tag": LLAMA2_ASSISTANT_TAG,
        "system": LLAMA2_SYSTEM,
        "system_prompt": LLAMA2_DEFAULT_SYSTEM_PROMPT,
        "sep_token": LLAMA2_SEP_TOKEN,
    },
    "llama3": {
        "user_tag": LLAMA3_USER_TAG,
        "assistant_tag": LLAMA3_ASSISTANT_TAG,
        "system": LLAMA3_SYSTEM,
        "sep_token": LLAMA3_SEP_TOKEN,
    },
}
