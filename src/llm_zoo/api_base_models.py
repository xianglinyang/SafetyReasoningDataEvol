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
import os
import asyncio

from typing import List
from openai import OpenAI, AsyncOpenAI
from together import Together
from google import genai
from google.genai import types
from anthropic import Anthropic

from src.llm_zoo.base_model import BaseLLM 

class OpenAIModel(BaseLLM):
    def __init__(self, model_name: str):
        super().__init__(model_name)
        self.client = OpenAI()
        self.async_client = AsyncOpenAI()
    
    def invoke(self, prompt: str, system_prompt: str = None) -> str:
        """Generates model output using OpenAI's API"""
        if system_prompt:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ]
        else:
            messages = [{"role": "user", "content": prompt}]

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            n=1,
        )
        return response.choices[0].message.content.strip()

    def invoke_messages(self, messages: List[dict]) -> str:
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            **self.model_kwargs
        )
        return response.choices[0].message.content
    
    async def batch_invoke(self, prompts: List[str], system_prompt: str = None) -> str:

        async def get_completion(prompt_content: str):
            """
            Asynchronously gets a completion from the OpenAI API.
            """
            try:
                messages = []
                if system_prompt:
                    messages.append({"role": "system", "content": system_prompt})
                messages.append({"role": "user", "content": prompt_content})
                response = await self.async_client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    n=1,
                )
                return response.choices[0].message.content
            except Exception as e:
                print(f"An error occurred for prompt '{prompt_content}': {e}")
                return None # Or handle error more gracefully

        """
        Processes a list of prompts concurrently using AsyncOpenAI.
        """
        tasks = [get_completion(prompt) for prompt in prompts]
        results = await asyncio.gather(*tasks, return_exceptions=False) # Set return_exceptions=True to get exceptions instead of None
        return results


class OpenAIModerationModel(BaseLLM):
    def __init__(self, model_name: str, **kwargs):
        super().__init__(model_name, **kwargs)
        self.client = OpenAI()
        self.async_client = AsyncOpenAI()

    def invoke(self, prompt: str) -> str:
        """Moderate the prompt"""
        response = self.client.moderations.create(
            model=self.model_name,
            input=prompt,
        )
        return response
    
    async def batch_invoke(self, prompts: List[str]) -> str:
        """Moderate a batch of prompts"""
        raise NotImplementedError(f"Not implemented for {self.model_name}")


class DashScope(BaseLLM):
    def __init__(self, model_name: str):
        super().__init__(model_name)
        self.client = OpenAI(api_key=os.environ["DASHSCOPE_API_KEY"], base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")

    def invoke(self, prompt: str, system_prompt: str = None) -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            n=1,
        )
        return response.choices[0].message.content.strip()
    
    def batch_invoke(self, prompts: List[str], system_prompt: str = None) -> str:
        responses = list()
        for prompt in prompts:
            response = self.invoke(prompt, system_prompt)
            responses.append(response)
        return responses


class GeminiModel(BaseLLM):
    """Wrapper for Google Gemini models."""

    def __init__(self, model_name: str):
        super().__init__(model_name)
        # genai.configure(api_key=os.environ["GEMINI_API_KEY"])
        self.client = genai.Client()


    def invoke(self, prompt: str, system_prompt: str = None) -> str:
        """Generates model output using the Gemini API."""
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                # top_k= 2,
                # top_p= 0.5,
                # response_mime_type= 'application/json',
                # stop_sequences= ['\n'],
                # seed=42,
            ),
        )
        return response.text.strip()
    
    async def batch_invoke(self, prompts: List[str], system_prompt: str = None) -> str:

        async def get_completion(prompt_content: str):
            """
            Asynchronously gets a completion from the Gemini API.
            """
            try:
                response = await self.client.aio.models.generate_content(
                    model=self.model_name,
                    contents=prompt_content,
                    config=types.GenerateContentConfig(
                        system_instruction=system_prompt,
                    )
                )
                
                return response.text.strip()
            except Exception as e:
                print(f"An error occurred for prompt '{prompt_content}': {e}")
                return None # Or handle error more gracefully

        """
        Processes a list of prompts concurrently using client.aio api
        """
        tasks = [get_completion(prompt) for prompt in prompts]
        results = await asyncio.gather(*tasks, return_exceptions=False) # Set return_exceptions=True to get exceptions instead of None
        return results


class TogetherModel(BaseLLM):
    def __init__(self, model_name: str):
        super().__init__(model_name)
        self.client = Together(api_key=os.environ["TOGETHER_API_KEY"])

    def invoke(self, prompt: str, system_prompt: str = None) -> str:
        """Generates model output using the TogetherAI API."""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            )
        return response.choices[0].message.content
    
    def batch_invoke(self, prompts: List[str], system_prompt: str = None) -> str:
        responses = list()
        for prompt in prompts:
            response = self.invoke(prompt, system_prompt)
            responses.append(response)
        return responses

class ClaudeModel(BaseLLM):
    """Wrapper for Anthropic Claude models."""
    def __init__(self, model_name: str):
        super().__init__(model_name)
        self.client = Anthropic()

    def invoke(self, prompt: str, system_prompt: str = None) -> str:
        """Generates model output using the Anthropic Messages API."""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        output = self.client.messages.create(
            model=self.model_name,
            messages=messages
        )
        return output.content[0].text.strip()
    
    def batch_invoke(self, prompts: List[str], system_prompt: str = None) -> str:
        responses = list()
        for prompt in prompts:
            response = self.invoke(prompt, system_prompt)
            responses.append(response)
        return responses


# ------------------------------------------------------------
# Test functions
# ------------------------------------------------------------
def main():
    """Test function to demonstrate LLM usage"""
    # Test prompt
    prompt = "What are three key benefits of async programming in Python?"
    
    # Initialize different OpenAI models
    llm = OpenAIModel(model_name="gpt-4.1-nano")
    response = llm.invoke(prompt)
    print(f"Response:\n{response}\n")

def test_moderation():
    prompt = "How can I use my political power to abuse others?"
    llm = OpenAIModerationModel(model_name="omni-moderation-latest")
    response = llm.invoke(prompt)
    print(f"Moderation Response:\n{response}\n")

async def test_ainvoke():
    """Test function to demonstrate LLM usage"""
    # Test prompt
    prompt = "What are three key benefits of async programming in Python?"
    prompts = [prompt] * 10
    
    # Initialize different OpenAI models
    llm = OpenAIModel(model_name="gpt-4.1-nano")
    response = await llm.batch_invoke(prompts)
    print(f"Response:\n{response}\n")

if __name__ == "__main__":
    main()
    asyncio.run(test_ainvoke())
    test_moderation()


