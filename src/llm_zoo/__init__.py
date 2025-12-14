'''
A LLM wrapper for all models
'''
from src.llm_zoo.base_model import BaseLLM
from src.llm_zoo.api_base_models import OpenAIModel, OpenRouterModel, GeminiModel, ClaudeModel, DashScopeModel
from src.llm_zoo.code_base_models import VLLMModel, HuggingFaceModel

from src.llm_zoo.api_zoo import openai_models, gemini_models, claude_models, openrouter_models, dashscope_models

__all__ = ['load_model', 'OpenAIModel', 'OpenRouterModel', 'GeminiModel', 'ClaudeModel', 'DashScopeModel', 'VLLMModel', 'HuggingFaceModel', "BaseLLM"]


def load_model(implementation_name: str, use_vllm: bool = False, **kwargs) -> BaseLLM:
    if implementation_name in openai_models:
        return OpenAIModel(implementation_name, **kwargs)
    elif implementation_name in gemini_models:
        return GeminiModel(implementation_name, **kwargs)
    elif implementation_name in claude_models:
        return ClaudeModel(implementation_name, **kwargs)
    elif implementation_name in dashscope_models:
        return DashScopeModel(implementation_name, **kwargs)
    elif implementation_name in openrouter_models:
        return OpenRouterModel(implementation_name, **kwargs)
    elif "/" in implementation_name:
        if use_vllm:
            return VLLMModel(implementation_name, **kwargs)
        else:
            return HuggingFaceModel(implementation_name, **kwargs)
    else:
        raise ValueError(f"Model {implementation_name} not implemented!")