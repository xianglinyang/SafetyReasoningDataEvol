# rough $ cost if pricing known (USD per 1M tokens)
from typing import NamedTuple

# Pricing information (USD per 1M tokens) - Update as needed
MODEL_PRICING = {
    # OpenAI models
    "gpt-4.1": {"input": 2.00, "output": 8.00},
    "gpt-4.1-mini": {"input": 0.40, "output": 1.60},
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4.1-nano": {"input": 0.10, "output": 0.40},
    "o4-mini": {"input": 1.10, "output": 4.40},
    "o3-mini": {"input": 1.00, "output": 4.00},
    "o3": {"input": 2.00, "output": 8.00},
    "gpt-5": {"input": 1.25, "output": 10.00},
    "gpt-5-mini": {"input": 0.25, "output": 2.00},
    "gpt-5-nano": {"input": 0.05, "output": 0.40},
    "gpt-5-chat-latest": {"input": 1.25, "output": 10.00},

    # Gemini models
    "gemini-2.5-pro": {"input": 1.25, "output": 10.00},
    "gemini-2.5-flash": {"input": 0.30, "output": 2.50},
    "gemini-2.5-flash-lite": {"input": 0.10, "output": 0.40},
    "gemini-2.5-flash-preview-05-20": {"input": 0.10, "output": 0.40},
    "gemini-2.5-pro-preview-06-05": {"input": 1.25, "output": 10.00},
    "gemini-2.5-pro-preview-03-25": {"input": 1.25, "output": 10.00},
    "gemini-2.0-flash-lite": {"input": 0.075, "output": 0.30},
    "gemini-2.0-flash": {"input": 0.10, "output": 0.40},
    "gemini-1.5-flash": {"input": 0.075, "output": 0.30},
    "gemini-1.5-flash-8b": {"input": 0.0375, "output": 0.15},
    "gemini-1.5-pro": {"input": 1.25, "output": 5.00},
    
    # Anthropic Models
    "claude-3-7-sonnet-20250219": {"input": 3.00, "output": 15.00},

    # OpenRouter OpenAI models
    "openai/gpt-5-mini": {"input": 0.25*1.05, "output": 2.00*1.05},
    "openai/gpt-5-nano": {"input": 0.05*1.05, "output": 0.40*1.05},
    "openai/gpt-5-chat-latest": {"input": 1.25*1.05, "output": 10.00*1.05},
    "openai/gpt-5": {"input": 1.25*1.05, "output": 10.00*1.05},
    "openai/o3-mini": {"input": 1.00*1.05, "output": 4.00*1.05},
    "openai/o3-mini-high": {"input": 1.00*1.05, "output": 4.00*1.05},
    "openai/o3": {"input": 2.00*1.05, "output": 8.00*1.05},
    "openai/o3-pro": {"input": 20.00*1.05, "output": 80.00*1.05}, # 1.05 is the tax rate
    "openai/o4-mini-high": {"input": 1.10*1.05, "output": 4.40*1.05},
    "openai/o4-mini": {"input": 1.10*1.05, "output": 4.40*1.05},
    "openai/gpt-4o": {"input": 2.50*1.05, "output": 10.00*1.05},
    "openai/gpt-4.1": {"input": 2.00*1.05, "output": 8.00*1.05},
    "openai/gpt-4.1-mini": {"input": 0.40*1.05, "output": 1.60*1.05},
    "openai/gpt-4.1-nano": {"input": 0.10*1.05, "output": 0.40*1.05},
    "openai/o1": {"input": 15.00*1.05, "output": 60.00*1.05},
    "openai/o1-mini": {"input": 1.00*1.05, "output": 4.00*1.05},
    "openai/chatgpt-4o-latest": {"input": 2.50*1.05, "output": 10.00*1.05},
    "openai/gpt-4o-mini": {"input": 0.15*1.05, "output": 0.60*1.05},
    "openai/gpt-4o": {"input": 2.50*1.05, "output": 10.00*1.05}, # 1.05 is the tax rate

    "google/gemini-2.5-flash-lite": {"input": 0.10*1.05, "output": 0.40*1.05},
    "google/gemini-2.5-flash-lite-preview-06-17": {"input": 0.10*1.05, "output": 0.40*1.05},
    "google/gemini-2.5-flash": {"input": 0.30*1.05, "output": 2.50*1.05},
    "google/gemini-2.5-pro": {"input": 1.25*1.05, "output": 10.00*1.05},
    "google/gemini-2.5-pro-preview": {"input": 1.25*1.05, "output": 10.00*1.05},
    "google/gemini-2.5-pro-preview-03-25": {"input": 1.25*1.05, "output": 10.00*1.05},

    "deepseek/deepseek-r1-0528": {"input": 0.18*1.05, "output": 0.72*1.05},
    "deepseek/deepseek-chat-v3-0324": {"input": 0.18*1.05, "output": 0.72*1.05},

    # OpenRouter Anthropic models
    "anthropic/claude-opus-4.1": {"input": 15.00*1.05, "output": 75.00*1.05},
    "anthropic/claude-opus-4": {"input": 15.00*1.05, "output": 75.00*1.05},
    "anthropic/claude-sonnet-4": {"input": 3.00*1.05, "output": 15.00*1.05},
    "anthropic/claude-3.7-sonnet": {"input": 3.00*1.05, "output": 15.00*1.05},
    "anthropic/claude-3.5-haiku": {"input": 0.80*1.05, "output": 4.00*1.05},
    "anthropic/claude-3.5-sonnet": {"input": 3.00*1.05, "output": 15.00*1.05},
    "anthropic/claude-3-haiku": {"input": 0.25*1.05, "output": 1.25*1.05},
    "anthropic/claude-3-opus": {"input": 15.00*1.05, "output": 75.00*1.05},

    # OpenRouter Qwen models
    "qwen/qwen-turbo": {"input": 0.05*1.05, "output": 0.20*1.05},
    "qwen/qwen-plus": {"input": 0.40*1.05, "output": 1.20*1.05},
    "qwen/qwen-max": {"input": 1.60*1.05, "output": 6.40*1.05},
    "qwen/qwen3-max": {"input": 1.20*1.05, "output": 6.00*1.05},
    "qwen/qwen3-235b-a22b-thinking-2507": {"input": 0.078*1.05, "output": 0.312*1.05},
    "qwen/qwen3-235b-a22b-2507": {"input": 0.078*1.05, "output": 0.312*1.05},
    "qwen/qwen3-30b-a3b-instruct-2507": {"input": 0.20*1.05, "output": 0.80*1.05},

    "meta-llama/llama-3.3-70b-instruct": {"input": 0.038*1.05, "output": 0.12*1.05},

    # X-AI
    "x-ai/grok-4": {"input": 3.00*1.05, "output": 15.00*1.05},
}

class CallResult(NamedTuple):
    """Result of an API call with cost information"""
    response: str
    input_tokens: int
    output_tokens: int
    cost: float
    model_name: str

def calculate_cost(model_name: str, input_tokens: int, output_tokens: int) -> float:
    """Calculate cost for a given model and token usage"""
    if model_name not in MODEL_PRICING:
        return 0.0
    
    pricing = MODEL_PRICING[model_name]
    input_cost = (input_tokens * pricing['input']) / 1_000_000.0
    output_cost = (output_tokens * pricing['output']) / 1_000_000.0
    return input_cost + output_cost

def estimate_tokens(text: str) -> int:
    """Estimate token count from text (rough approximation)"""
    # Rough approximation: 1 token ≈ 4 characters for English text
    return max(1, len(text) // 4)
    