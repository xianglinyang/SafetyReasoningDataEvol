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
from typing import Optional, Dict, Any

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
    
    CHAT_MODELS = {"gpt-4", "gpt-3.5-turbo", "gpt-4-turbo-preview"}
    INSTRUCT_MODELS = {"gpt-3.5-turbo-instruct", "text-davinci-003"}
    
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
                messages=[{"role": "user", "content": prompt}],
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


# class ClaudeLLM(BaseLLM):
#     def __init__(self, model_name: str = "claude-3-sonnet", **kwargs):
#         super().__init__(model_name, **kwargs)
#         from anthropic import AsyncAnthropic
#         self.client = AsyncAnthropic()

#     async def generate(self, prompt: str) -> str:
#         response = await self.client.messages.create(
#             model=self.model_name,
#             messages=[{"role": "user", "content": prompt}],
#             **self.model_kwargs
#         )
#         return response.content[0].text

# class GeminiLLM(BaseLLM):
#     def __init__(self, model_name: str = "gemini-pro", **kwargs):
#         super().__init__(model_name, **kwargs)
#         import google.generativeai as genai
#         self.client = genai

#     async def generate(self, prompt: str) -> str:
#         model = self.client.GenerativeModel(self.model_name)
#         response = await model.generate_content_async(prompt)
#         return response.text

# class QwenLLM(BaseLLM):
#     def __init__(self, model_name: str = "qwen-max", **kwargs):
#         super().__init__(model_name, **kwargs)
#         from dashscope import Generation
#         self.client = Generation()

#     async def generate(self, prompt: str) -> str:
#         response = await self.client.call(
#             model=self.model_name,
#             prompt=prompt,
#             **self.model_kwargs
#         )
#         return response.output.text



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

    # # Test each model
    # for name, llm in llms.items():
    #     try:
    #         print(f"\nTesting {name}...")
    #         response = await llm.generate(prompt)
    #         print(f"{name} Response:\n{response}\n")
    #         print("-" * 50)
    #     except Exception as e:
    #         print(f"Error with {name}: {str(e)}")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
