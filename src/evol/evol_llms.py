"""
Define the LLM for evolving the dataset.
1. openai model
2. claude model
3. gemini model
4. qwen model

In need of:
   - `OPENAI_API_KEY`
   - `ANTHROPIC_API_KEY`
   - `GOOGLE_API_KEY`
   - `DASHSCOPE_API_KEY`

Use openai for now.
"""

from abc import ABC, abstractmethod

SYSTEM_PROMPT = "You are a helpful, respectful and honest assistant."

class BaseLLM(ABC):
    """Base class for LLM wrappers"""
    
    def __init__(self, model_name: str, **kwargs):
        self.model_name = model_name
        self.model_kwargs = kwargs

    @abstractmethod
    async def generate(self, prompt: str) -> str:
        """Generate response from the LLM"""
        pass

class OpenAILLM(BaseLLM):
    """OpenAI LLM wrapper that supports both chat and instruct models"""
    
    # GPT-4o models
    GPT4O_MODELS = {
        "gpt-4o",                          # Latest flagship model
        "gpt-4o-2024-11-20",              # November snapshot
        "gpt-4o-2024-08-06",              # August snapshot with Structured Outputs
        "gpt-4o-2024-05-13",              # Original snapshot
        "chatgpt-4o-latest",              # Dynamic ChatGPT version
    }
    
    # GPT-4o mini models
    GPT4O_MINI_MODELS = {
        "gpt-4o-mini",                     # Latest mini model
        "gpt-4o-mini-2024-07-18",         # July snapshot
    }
    
    # GPT-4 Turbo and GPT-4 models
    GPT4_MODELS = {
        "gpt-4-turbo",                     # Latest GPT-4 Turbo with vision
        "gpt-4-turbo-2024-04-09",         # April snapshot
        "gpt-4-turbo-preview",             # Preview version
        "gpt-4-0125-preview",             # January preview
        "gpt-4-1106-preview",             # November preview
        "gpt-4",                          # Standard GPT-4
        "gpt-4-0613",                     # June snapshot
        "gpt-4-0314",                     # March snapshot
    }
    
    # GPT-3.5 Turbo models
    GPT35_MODELS = {
        "gpt-3.5-turbo",                  # Latest version
        "gpt-3.5-turbo-0125",             # January snapshot
        "gpt-3.5-turbo-1106",             # November snapshot
        "gpt-3.5-turbo-instruct",         # Instruct version
    }
    
    # Combined sets for API usage
    CHAT_MODELS = GPT4O_MODELS | GPT4O_MINI_MODELS | GPT4_MODELS | GPT35_MODELS - {"gpt-3.5-turbo-instruct"}
    INSTRUCT_MODELS = {"gpt-3.5-turbo-instruct"}
    
    def __init__(self, model_name: str = "gpt-4", **kwargs):
        super().__init__(model_name, **kwargs)
        from openai import AsyncOpenAI
        self.client = AsyncOpenAI()
        
        if model_name not in self.CHAT_MODELS and model_name not in self.INSTRUCT_MODELS:
            raise ValueError(f"Unknown model: {model_name}. Must be one of {self.CHAT_MODELS | self.INSTRUCT_MODELS}")

    async def generate(self, prompt: str) -> str:
        """Generate response using either chat or instruct model based on model_name"""
        if self.model_name in self.CHAT_MODELS:
            response = await self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt}
                    ],
                **self.model_kwargs
            )
            return response.choices[0].message.content
        else:  # instruct model
            response = await self.client.completions.create(
                model=self.model_name,
                prompt=prompt,
                **self.model_kwargs
            )
            return response.choices[0].text.strip()

LLM_MODELS = {
    "OPENAI_LLM": OpenAILLM.CHAT_MODELS | OpenAILLM.INSTRUCT_MODELS,
}

async def llm_generate(model_name: str, prompt: str, **kwargs) -> str:
    if model_name in LLM_MODELS["OPENAI_LLM"]:
        llm = OpenAILLM(model_name=model_name, **kwargs)
    else:
        raise ValueError(f"Unknown model: {model_name}. Must be one of {LLM_MODELS['OPENAI_LLM']}")
    response = await llm.generate(prompt)
    return response


async def main():
    """Test function to demonstrate LLM usage"""
    # Test prompt
    prompt = "What are three key benefits of async programming in Python?"
    
    # Initialize different OpenAI models
    llms = {
        "GPT-4": OpenAILLM(model_name="gpt-4", temperature=0.7, max_tokens=150),
        "GPT-3.5-Turbo": OpenAILLM(model_name="gpt-3.5-turbo", temperature=0.7, max_tokens=150),
        "GPT-3.5-Turbo-Instruct": OpenAILLM(model_name="gpt-3.5-turbo-instruct", temperature=0.7, max_tokens=150),
    }
    name = "GPT-3.5-Turbo-Instruct"
    llm = llms[name]
    response = await llm.generate(prompt)
    print(f"{name} Response:\n{response}\n")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
