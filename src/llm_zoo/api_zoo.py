'''
This file is used to map the implementation name to the name of the model.
'''

import requests
import logging

logger = logging.getLogger(__name__)

# Implementation names
openai_models = [
    "gpt-4.1",
    "gpt-4.1-mini",
    "gpt-4o",
    "gpt-4o-mini",
    "gpt-4.1-nano",
    "o4-mini",
    "o3-mini",
    "o3",
    "gpt-5",
    "gpt-5-mini",
    "gpt-5-nano",
    "gpt-5-chat-latest",
]

gemini_models = [
    "gemini-2.5-pro",
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
    "gemini-2.5-flash-preview-05-20",
    "gemini-2.5-pro-preview-06-05",
    "gemini-2.5-pro-preview-03-25",
    "gemini-2.0-flash-lite",
    "gemini-2.0-flash",
    "gemini-1.5-flash",
    "gemini-1.5-flash-8b",
    "gemini-1.5-pro",
]

claude_models = [
    "claude-3-7-sonnet-20250219",
]

openrouter_models = [
    # openai
    "openai/gpt-5-mini",
    "openai/gpt-5-nano",
    "openai/gpt-5-chat-latest",
    "openai/gpt-5",
    "openai/o3-mini",
    "openai/o3-mini-high",
    "openai/o3",
    "openai/o3-pro",
    "openai/o4-mini-high",
    "openai/o4-mini",
    "openai/gpt-4.1",
    "openai/gpt-4.1-mini",
    "openai/gpt-4.1-nano",
    "openai/gpt-4o",
    "openai/o1",
    "openai/o1-mini",
    "openai/chatgpt-4o-latest",
    "openai/gpt-4o-mini",
    "openai/gpt-4-turbo",
    "openai/gpt-3.5-turbo",

    # gemini
    "google/gemini-2.5-flash-lite",
    "google/gemini-2.5-flash-lite-preview-06-17",
    "google/gemini-2.5-flash",
    "google/gemini-2.5-pro",
    "google/gemini-2.5-pro-preview",
    "google/gemini-2.5-pro-preview-03-25",
    "google/gemini-2.0-flash-lite-001",
    "google/gemini-2.0-flash",

    # DeepSeek
    "deepseek/deepseek-r1-0528",
    "deepseek/deepseek-chat-v3-0324",
    "deepseek/deepseek-r1-0528-qwen3-8b",

    # anthropic
    "anthropic/claude-opus-4.1",
    "anthropic/claude-opus-4",
    "anthropic/claude-sonnet-4",
    "anthropic/claude-3.7-sonnet",
    "anthropic/claude-3.5-haiku",
    "anthropic/claude-3.5-sonnet",
    "anthropic/claude-3-haiku",
    "anthropic/claude-3-opus",

    # qwen
    # 3.0
    "qwen/qwen3-max",
    "qwen/qwen3-235b-a22b-thinking-2507",
    "qwen/qwen3-235b-a22b-2507",
    "qwen/qwen3-30b-a3b-instruct-2507",
    "qwen/qwen3-8b",

    # 2.5
    "qwen/qwen-turbo",
    "qwen/qwen-plus",
    "qwen/qwen-max",

    # Meta Llama
    "meta-llama/llama-4-scout",
    "meta-llama/llama-3.3-70b-instruct",
    "meta-llama/llama-3.1-70b-instruct",
    "meta-llama/llama-3-70b-instruct",
    "meta-llama/llama-3.1-405b-instruct",
    "meta-llama/llama-3-8b-instruct",

    # X-AI
    "x-ai/grok-4",
]

dashscope_models = [
    "deepseek-r1",
    "deepseek-v3",
    "deepseek-r1-distill-qwen-7b",
    "deepseek-r1-distill-qwen-14b",
    "deepseek-r1-distill-qwen-32b",
    "deepseek-r1-distill-llama-8b",
    "deepseek-r1-distill-llama-70b",

    # 3.0
    "qwen-plus",
    "qwen-turbo",
    "qwen-max",
    "qwen3-235b-a22b-thinking-2507",
    "qwen3-235b-a22b-instruct-2507",
    "qwen3-235b-a22b",
    "qwen3-32b",
    "qwen3-30b-a3b-thinking-2507",
    "qwen3-30b-a3b-instruct-2507",
    "qwen3-30b-a3b",
    "qwen3-14b",
    "qwen3-8b",

    # 2.5
    "qwq-plus",
    "qwq-32b"
]

# mapping
implementation_to_name = {
    # OpenAI models
    "gpt-4.1": "gpt-4.1",
    "gpt-4.1-mini": "gpt-4.1-mini",
    "gpt-4o-mini": "gpt-4o-mini",
    "gpt-4o": "gpt-4o",
    "gpt-4.1-nano": "gpt-4.1-nano",
    "o4-mini": "o4-mini",
    "o3-mini": "o3-mini",
    "o3": "o3",
    "gpt-5": "gpt-5",
    "gpt-5-mini": "gpt-5-mini",
    "gpt-5-nano": "gpt-5-nano",
    "gpt-5-chat-latest": "gpt-5-chat-latest",

    # Gemini models
    "gemini-2.5-pro": "gemini-2.5-pro",
    "gemini-2.5-flash": "gemini-2.5-flash",
    "gemini-2.5-flash-lite-preview-06-17": "gemini-2.5-flash-lite-preview-06-17",
    "gemini-2.5-flash-preview-05-20": "gemini-2.5-flash-preview-05-20",
    "gemini-2.5-pro-preview-06-05": "gemini-2.5-pro-preview-06-05",
    "gemini-2.5-pro-preview-03-25": "gemini-2.5-pro-preview-03-25",
    "gemini-2.0-flash-lite": "gemini-2.0-flash-lite",
    "gemini-2.0-flash": "gemini-2.0-flash",
    "gemini-1.5-flash": "gemini-1.5-flash",
    "gemini-1.5-flash-8b": "gemini-1.5-flash-8b",
    "gemini-1.5-pro": "gemini-1.5-pro",

    # Claude models
    "claude-3-7-sonnet-20250219": "claude-3-7-sonnet-20250219",

    # OpenRouter OpenAI models
    "openai/gpt-5-mini": "gpt-5-mini",
    "openai/gpt-5-nano": "gpt-5-nano",
    "openai/gpt-5-chat-latest": "gpt-5-chat-latest",
    "openai/gpt-5": "gpt-5",
    "openai/o3-mini": "o3-mini",
    "openai/o3-mini-high": "o3-mini-high",
    "openai/o3": "o3",
    "openai/o3-pro": "o3-pro",
    "openai/o4-mini-high": "o4-mini-high",
    "openai/o4-mini": "o4-mini",
    "openai/gpt-4.1": "gpt-4.1",
    "openai/gpt-4.1-mini": "gpt-4.1-mini",
    "openai/gpt-4.1-nano": "gpt-4.1-nano",
    "openai/o1": "o1",
    "openai/o1-mini": "o1-mini",
    "openai/chatgpt-4o-latest": "chatgpt-4o-latest",
    "openai/gpt-4o-mini": "gpt-4o-mini",
    "openai/gpt-4o": "gpt-4o",
    "openai/gpt-4-turbo": "gpt-4-turbo",
    "openai/gpt-3.5-turbo": "gpt-3.5-turbo",

    # OpenRouter Gemini models
    "google/gemini-2.5-flash-lite": "gemini-2.5-flash-lite",
    "google/gemini-2.5-flash-lite-preview-06-17": "gemini-2.5-flash-lite-preview-06-17",
    "google/gemini-2.5-flash": "gemini-2.5-flash",
    "google/gemini-2.5-pro": "gemini-2.5-pro",
    "google/gemini-2.5-pro-preview": "gemini-2.5-pro-preview",
    "google/gemini-2.5-pro-preview-03-25": "gemini-2.5-pro-preview-03-25",
    "google/gemini-2.0-flash-lite-001": "gemini-2.0-flash-lite",
    "google/gemini-2.0-flash": "gemini-2.0-flash",

    # OpenRouter DeepSeek models
    "deepseek/deepseek-r1-0528": "deepseek-r1-0528",
    "deepseek/deepseek-chat-v3-0324": "deepseek-chat-v3-0324",
    "deepseek/deepseek-r1-0528-qwen3-8b": "deepseek-r1-0528-qwen3-8b",
    "deepseek/deepseek-r1-distill-qwen-7b": "deepseek-r1-distill-qwen-7b",

    # OpenRouter Anthropic models
    "anthropic/claude-opus-4.1": "claude-opus-4.1",
    "anthropic/claude-opus-4": "claude-opus-4",
    "anthropic/claude-sonnet-4": "claude-sonnet-4",
    "anthropic/claude-3.7-sonnet": "claude-3.7-sonnet",
    "anthropic/claude-3.5-haiku": "claude-3.5-haiku",
    "anthropic/claude-3.5-sonnet": "claude-3.5-sonnet",
    "anthropic/claude-3-haiku": "claude-3-haiku",
    "anthropic/claude-3-opus": "claude-3-opus",

    # OpenRouter Qwen models
    "qwen/qwen-turbo": "qwen-turbo",
    "qwen/qwen-plus": "qwen-plus",
    "qwen/qwen-max": "qwen-max",
    "qwen/qwen3-max": "qwen3-max",
    "qwen/qwen3-235b-a22b-thinking-2507": "qwen3-235b-a22b-thinking-2507",
    "qwen/qwen3-235b-a22b-2507": "qwen3-235b-a22b-2507",
    "qwen/qwen3-30b-a3b-instruct-2507": "qwen3-30b-a3b-instruct-2507",
    "qwen/qwen3-8b": "qwen3-8b",

    # OpenRouter Meta Llama models
    "meta-llama/llama-4-scout": "llama-4-scout",
    "meta-llama/llama-3.3-70b-instruct": "llama-3.3-70b-instruct",
    "meta-llama/llama-3.1-70b-instruct": "llama-3.1-70b-instruct",
    "meta-llama/llama-3-70b-instruct": "llama-3-70b-instruct",
    "meta-llama/llama-3.1-405b-instruct": "llama-3.1-405b-instruct",
    "meta-llama/llama-3-8b-instruct": "llama-3-8b-instruct",

    # DashScope DeepSeek models
    "deepseek-r1": "deepseek-r1",
    "deepseek-v3": "deepseek-v3",
    "deepseek-r1-distill-qwen-7b": "deepseek-r1-distill-qwen-7b",
    "deepseek-r1-distill-qwen-14b": "deepseek-r1-distill-qwen-14b",
    "deepseek-r1-distill-qwen-32b": "deepseek-r1-distill-qwen-32b",
    "deepseek-r1-distill-llama-8b": "deepseek-r1-distill-llama-8b",
    "deepseek-r1-distill-llama-70b": "deepseek-r1-distill-llama-70b",

    # DashScope Qwen models
    "qwen-plus": "qwen-plus",
    "qwen-turbo": "qwen-turbo",
    "qwen-max": "qwen-max",
    "qwen3-235b-a22b-thinking-2507": "qwen3-235b-a22b-thinking-2507",
    "qwen3-235b-a22b-instruct-2507": "qwen3-235b-a22b-instruct-2507",
    "qwen3-235b-a22b": "qwen3-235b-a22b",
    "qwen3-32b": "qwen3-32b",
    "qwen3-max": "qwen3-max",
    "qwen3-30b-a3b-thinking-2507": "qwen3-30b-a3b-thinking-2507",
    "qwen3-30b-a3b-instruct-2507": "qwen3-30b-a3b-instruct-2507",
    "qwen3-30b-a3b": "qwen3-30b-a3b",
    "qwen3-14b": "qwen3-14b",
    "qwen3-8b": "qwen3-8b",
    "qwq-plus": "qwq-plus",
    "qwq-32b": "qwq-32b",

    # OpenRouter X-AI models
    "x-ai/grok-4": "grok-4",
}

def is_huggingface_model(model_name: str) -> bool:
    """
    Check if a model exists on Hugging Face Hub.
    
    Args:
        model_name (str): The model name to check (e.g., "microsoft/DialoGPT-medium")
    
    Returns:
        bool: True if the model exists on Hugging Face Hub, False otherwise
    """
    try:
        # Use Hugging Face Hub API to check model existence
        url = f"https://huggingface.co/api/models/{model_name}"
        response = requests.get(url, timeout=10)
        
        if response.status_code == 200:
            return True
        elif response.status_code == 404:
            return False
        else:
            # For other status codes, we'll be conservative and return False
            return False
            
    except requests.RequestException:
        # If there's a network error, we'll be conservative and return False
        return False
    except Exception:
        # For any other exception, return False
        return False


def get_model_name(implementation_name):
    """
    Get the model name from implementation name.
    
    Args:
        implementation_name (str): The implementation name of the model
    
    Returns:
        str: The model name, or the original name if not found in mapping
    """
    if implementation_name in implementation_to_name.keys():
        return implementation_to_name.get(implementation_name, implementation_name)
    elif is_huggingface_model(implementation_name):
        return implementation_name.split("/")[-1]
    else:
        # logger.warning(f"Implementation name {implementation_name} is not valid")
        return implementation_name
        # raise ValueError(f"Implementation name {implementation_name} is not valid")

def is_valid_model(implementation_name):
    """
    Check if an implementation name is valid.
    
    Args:
        implementation_name (str): The implementation name to check
    
    Returns:
        bool: True if the implementation name is valid, False otherwise
    """
    if implementation_name in implementation_to_name.keys():
        logger.info(f"Implementation name {implementation_name} is valid as an API based model")
        return True
    
    if is_huggingface_model(implementation_name):
        logger.info(f"Implementation name {implementation_name} is valid as a Hugging Face model")
        return True
    
    return False

def get_all_models():
    """
    Get all model names.
    
    Returns:
        list: List of all model names
    """
    return list(set(implementation_to_name.values()))

def get_all_implementation_names():
    """
    Get all implementation names.
    
    Returns:
        list: List of all implementation names
    """
    return list(implementation_to_name.keys())


if __name__ == "__main__":
    name1 = "openai/gpt-4.1-mini"
    name2 = "google/gemini-2.5-flash"
    name3 = "qwen3-14b"

    c1 = is_valid_model(name1)
    c2 = is_valid_model(name2)
    c3 = is_valid_model(name3)

    print(c1, c2, c3)
    
    
