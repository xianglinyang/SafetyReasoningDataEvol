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
from src.llm_zoo.rate_limiter import rate_limited_async_call, OPENAI_RATE_LIMIT, GEMINI_RATE_LIMIT


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
    
    @rate_limited_async_call(OPENAI_RATE_LIMIT)
    async def _get_completion(self, prompt_content: str, system_prompt: str = None):
        """
        Asynchronously gets a completion from the OpenAI API with rate limiting.
        """
        max_retries = 3
        retry_delay = 2.0
        
        for attempt in range(max_retries):
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
                if attempt < max_retries - 1:
                    print(f"Attempt {attempt + 1} failed for prompt '{prompt_content[:50]}...': {e}")
                    await asyncio.sleep(retry_delay * (2 ** attempt))  # Exponential backoff
                else:
                    print(f"All retries failed for prompt '{prompt_content[:50]}...': {e}")
                    return None
    
    async def batch_invoke(self, prompts: List[str], system_prompt: str = None, batch_size: int = 5000, delay_between_batches: float = 1.0) -> List[str]:
        """
        Processes a list of prompts in batches with rate limiting to avoid overwhelming the API.
        
        Args:
            prompts: List of prompts to process
            system_prompt: Optional system prompt
            batch_size: Number of prompts to process in each batch (default: 50)
            delay_between_batches: Delay in seconds between batches (default: 1.0)
        """
        all_results = []
        total_batches = (len(prompts) + batch_size - 1) // batch_size
        
        print(f"Processing {len(prompts)} prompts in {total_batches} batches of size {batch_size}")
        
        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i:i + batch_size]
            batch_num = i // batch_size + 1
            
            print(f"Processing batch {batch_num}/{total_batches} ({len(batch_prompts)} prompts)")
            
            # Process current batch with limited concurrency
            tasks = [self._get_completion(prompt, system_prompt) for prompt in batch_prompts]
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Handle exceptions in batch results
            processed_results = []
            for j, result in enumerate(batch_results):
                if isinstance(result, Exception):
                    print(f"Exception in batch {batch_num}, prompt {j}: {result}")
                    processed_results.append(None)
                else:
                    processed_results.append(result)
            
            all_results.extend(processed_results)
            
            # Add delay between batches (except for the last batch)
            if i + batch_size < len(prompts):
                print(f"Waiting {delay_between_batches} seconds before next batch...")
                await asyncio.sleep(delay_between_batches)
        
        print(f"Completed processing all {len(prompts)} prompts")
        return all_results


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
    
    @rate_limited_async_call(GEMINI_RATE_LIMIT)
    async def _get_completion(self, prompt_content: str, system_prompt: str = None):
        """
        Asynchronously gets a completion from the Gemini API with rate limiting.
        """
        max_retries = 3
        retry_delay = 2.0
        
        for attempt in range(max_retries):
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
                if attempt < max_retries - 1:
                    print(f"Attempt {attempt + 1} failed for prompt '{prompt_content[:50]}...': {e}")
                    await asyncio.sleep(retry_delay * (2 ** attempt))  # Exponential backoff
                else:
                    print(f"All retries failed for prompt '{prompt_content[:50]}...': {e}")
                    return None
    
    async def batch_invoke(self, prompts: List[str], system_prompt: str = None, batch_size: int = 200, delay_between_batches: float = 1.0) -> List[str]:
        """
        Processes a list of prompts in batches with rate limiting to avoid overwhelming the API.
        
        Args:
            prompts: List of prompts to process
            system_prompt: Optional system prompt
            batch_size: Number of prompts to process in each batch (default: 50)
            delay_between_batches: Delay in seconds between batches (default: 1.0)
        """
        all_results = []
        total_batches = (len(prompts) + batch_size - 1) // batch_size
        
        print(f"Processing {len(prompts)} prompts in {total_batches} batches of size {batch_size}")
        
        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i:i + batch_size]
            batch_num = i // batch_size + 1
            
            print(f"Processing batch {batch_num}/{total_batches} ({len(batch_prompts)} prompts)")
            
            # Process current batch with limited concurrency
            tasks = [self._get_completion(prompt, system_prompt) for prompt in batch_prompts]
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Handle exceptions in batch results
            processed_results = []
            for j, result in enumerate(batch_results):
                if isinstance(result, Exception):
                    print(f"Exception in batch {batch_num}, prompt {j}: {result}")
                    processed_results.append(None)
                else:
                    processed_results.append(result)
            
            all_results.extend(processed_results)
            
            # Add delay between batches (except for the last batch)
            if i + batch_size < len(prompts):
                print(f"Waiting {delay_between_batches} seconds before next batch...")
                await asyncio.sleep(delay_between_batches)
        
        print(f"Completed processing all {len(prompts)} prompts")
        return all_results


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


