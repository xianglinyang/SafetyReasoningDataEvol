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
import logging
from dotenv import load_dotenv
load_dotenv()

from src.llm_zoo.base_model import BaseLLM 
from src.llm_zoo.rate_limiter import rate_limited_async_call, OPENAI_RATE_LIMIT, GEMINI_RATE_LIMIT, OPENROUTER_RATE_LIMIT
from src.llm_zoo.cost_utils import calculate_cost, estimate_tokens, MODEL_PRICING, CallResult

logger = logging.getLogger(__name__)


class OpenAIModel(BaseLLM):
    def __init__(self, model_name: str):
        super().__init__(model_name)
        self.client = OpenAI()
        self.async_client = AsyncOpenAI()
    
    def invoke(self, prompt: str, system_prompt: str = None, return_cost: bool = False) -> CallResult:
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
        
        if return_cost:
            # Calculate costs
            input_tokens = response.usage.prompt_tokens
            output_tokens = response.usage.completion_tokens
            cost = calculate_cost(self.model_name, input_tokens, output_tokens)
            
            return CallResult(
                response=response.choices[0].message.content.strip(),
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cost=cost,
                model_name=self.model_name
            )
        else:
            return response.choices[0].message.content.strip()

    def invoke_messages(self, messages: List[dict], return_cost: bool = False) -> CallResult:
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            **self.model_kwargs
        )
        
        if return_cost:
            # Calculate costs
            input_tokens = response.usage.prompt_tokens
            output_tokens = response.usage.completion_tokens
            cost = calculate_cost(self.model_name, input_tokens, output_tokens)
            
            return CallResult(
                response=response.choices[0].message.content,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cost=cost,
                model_name=self.model_name
            )
        else:
            return response.choices[0].message.content
    
    @rate_limited_async_call(OPENAI_RATE_LIMIT)
    async def _get_completion(self, prompt_content: str, system_prompt: str = None, return_cost: bool = False) -> CallResult:
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
                if return_cost:
                    # Calculate costs
                    input_tokens = response.usage.prompt_tokens
                    output_tokens = response.usage.completion_tokens
                    cost = calculate_cost(self.model_name, input_tokens, output_tokens)
                    
                    return CallResult(
                        response=response.choices[0].message.content,
                        input_tokens=input_tokens,
                        output_tokens=output_tokens,
                        cost=cost,
                        model_name=self.model_name
                    )
                else:
                    return response.choices[0].message.content
            except Exception as e:
                if attempt < max_retries - 1:
                    print(f"Attempt {attempt + 1} failed for prompt '{prompt_content[:50]}...': {e}")
                    await asyncio.sleep(retry_delay * (2 ** attempt))  # Exponential backoff
                else:
                    print(f"All retries failed for prompt '{prompt_content[:50]}...': {e}")
                    return None
    
    async def batch_invoke(self, prompts: List[str], system_prompt: str = None, return_cost: bool = False) -> List[CallResult]:
        """
        Processes a list of prompts in batches with rate limiting to avoid overwhelming the API.
        
        Args:
            prompts: List of prompts to process
            system_prompt: Optional system prompt
            batch_size: Number of prompts to process in each batch (default: 50)
            delay_between_batches: Delay in seconds between batches (default: 1.0)
        """
        all_results = []

        # Process current batch with limited concurrency
        tasks = [self._get_completion(prompt, system_prompt, return_cost) for prompt in prompts]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle exceptions in batch results
        for j, result in enumerate(results):
            if isinstance(result, Exception):
                all_results.append(None)
            else:
                all_results.append(result)
        
        return all_results


class OpenRouterModel(BaseLLM):
    def __init__(self, model_name: str):
        super().__init__(model_name)
        # rewrite the client and async_client
        self.client = OpenAI(api_key=os.environ["OPENROUTER_API_KEY"], base_url="https://openrouter.ai/api/v1")
        self.async_client = AsyncOpenAI(api_key=os.environ["OPENROUTER_API_KEY"], base_url="https://openrouter.ai/api/v1")
    
    def invoke(self, prompt: str, system_prompt: str = None, return_cost: bool = False) -> CallResult:
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

        if return_cost:
        
            # Calculate costs
            input_tokens = response.usage.prompt_tokens
            output_tokens = response.usage.completion_tokens
            cost = calculate_cost(self.model_name, input_tokens, output_tokens)
            
            return CallResult(
                response=response.choices[0].message.content.strip(),
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cost=cost,
                model_name=self.model_name
            )
        else:
            return response.choices[0].message.content.strip()

    def invoke_messages(self, messages: List[dict], return_cost: bool = False) -> CallResult:
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            **self.model_kwargs
        )
        if return_cost:
            # Calculate costs
            input_tokens = response.usage.prompt_tokens
            output_tokens = response.usage.completion_tokens
            cost = calculate_cost(self.model_name, input_tokens, output_tokens)
            
            return CallResult(
                response=response.choices[0].message.content,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cost=cost,
                model_name=self.model_name
            )
        else:
            return response.choices[0].message.content
    
    @rate_limited_async_call(OPENROUTER_RATE_LIMIT)
    async def _get_completion(self, prompt_content: str, system_prompt: str = None, return_cost: bool = False) -> CallResult:
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
                
                if return_cost:
                    # Calculate costs
                    input_tokens = response.usage.prompt_tokens
                    output_tokens = response.usage.completion_tokens
                    cost = calculate_cost(self.model_name, input_tokens, output_tokens)
                    
                    return CallResult(
                        response=response.choices[0].message.content,
                        input_tokens=input_tokens,
                        output_tokens=output_tokens,
                        cost=cost,
                        model_name=self.model_name
                    )
                else:
                    return response.choices[0].message.content
            except Exception as e:
                if attempt < max_retries - 1:
                    print(f"Attempt {attempt + 1} failed for prompt '{prompt_content[:50]}...': {e}")
                    await asyncio.sleep(retry_delay * (2 ** attempt))  # Exponential backoff
                else:
                    print(f"All retries failed for prompt '{prompt_content[:50]}...': {e}")
                    return None
    
    async def batch_invoke(self, prompts: List[str], system_prompt: str = None, return_cost: bool = False) -> List[CallResult]:
        """
        Processes a list of prompts in batches with rate limiting to avoid overwhelming the API.
        
        Args:
            prompts: List of prompts to process
            system_prompt: Optional system prompt
            batch_size: Number of prompts to process in each batch (default: 50)
            delay_between_batches: Delay in seconds between batches (default: 1.0)
        """
        all_results = []
        
        # Process current batch with limited concurrency
        tasks = [self._get_completion(prompt, system_prompt, return_cost) for prompt in prompts]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle exceptions in batch results
        for j, result in enumerate(results):
            if isinstance(result, Exception):
                all_results.append(None)
            else:
                all_results.append(result)
        
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


class DashScopeModel(BaseLLM):
    def __init__(self, model_name: str):
        super().__init__(model_name)
        self.client = OpenAI(api_key=os.environ["DASHSCOPE_API_KEY"], base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")
        self.async_client = AsyncOpenAI(api_key=os.environ["DASHSCOPE_API_KEY"], base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")
    
    def invoke(self, prompt: str, system_prompt: str = None, return_cost: bool = False) -> CallResult:
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
        if return_cost:
        
            # Calculate costs
            input_tokens = response.usage.prompt_tokens
            output_tokens = response.usage.completion_tokens
            cost = calculate_cost(self.model_name, input_tokens, output_tokens)
            
            return CallResult(
                response=response.choices[0].message.content.strip(),
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cost=cost,
                model_name=self.model_name
            )
        else:
            return response.choices[0].message.content.strip()

    def invoke_messages(self, messages: List[dict]) -> str:
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            **self.model_kwargs
        )
        return response.choices[0].message.content
    
    @rate_limited_async_call(OPENAI_RATE_LIMIT)
    async def _get_completion(self, prompt_content: str, system_prompt: str = None, return_cost: bool = False) -> CallResult:
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
                
                if return_cost:
                    # Calculate costs
                    input_tokens = response.usage.prompt_tokens
                    output_tokens = response.usage.completion_tokens
                    cost = calculate_cost(self.model_name, input_tokens, output_tokens)
                    
                    return CallResult(
                        response=response.choices[0].message.content,
                        input_tokens=input_tokens,
                        output_tokens=output_tokens,
                        cost=cost,
                        model_name=self.model_name
                    )
                else:
                    return response.choices[0].message.content
            except Exception as e:
                if attempt < max_retries - 1:
                    print(f"Attempt {attempt + 1} failed for prompt '{prompt_content[:50]}...': {e}")
                    await asyncio.sleep(retry_delay * (2 ** attempt))  # Exponential backoff
                else:
                    print(f"All retries failed for prompt '{prompt_content[:50]}...': {e}")
                    return None
    
    async def batch_invoke(self, prompts: List[str], system_prompt: str = None, return_cost: bool = False) -> List[CallResult]:
        """
        Processes a list of prompts in batches with rate limiting to avoid overwhelming the API.
        
        Args:
            prompts: List of prompts to process
            system_prompt: Optional system prompt
            batch_size: Number of prompts to process in each batch (default: 50)
            delay_between_batches: Delay in seconds between batches (default: 1.0)
        """
        all_results = []

        # Process current batch with limited concurrency
        tasks = [self._get_completion(prompt, system_prompt, return_cost) for prompt in prompts]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle exceptions in batch results
        for j, result in enumerate(results):
            if isinstance(result, Exception):
                all_results.append(None)
            else:
                all_results.append(result)
        
        return all_results
    

class GeminiModel(BaseLLM):
    """Wrapper for Google Gemini models."""

    def __init__(self, model_name: str):
        super().__init__(model_name)
        # genai.configure(api_key=os.environ["GEMINI_API_KEY"])
        self.client = genai.Client()

    def invoke(self, prompt: str, system_prompt: str = None, return_cost: bool = False) -> CallResult:
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
        if return_cost:
            # Calculate costs (estimate tokens since Gemini doesn't provide usage info)
            input_tokens = response.usage_metadata.prompt_token_count
            output_tokens = response.usage_metadata.candidates_token_count
            cost = calculate_cost(self.model_name, input_tokens, output_tokens)
            
            return CallResult(
                response=response.text.strip(),
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cost=cost,
                model_name=self.model_name
            )
        else:
            return response.text.strip()
    
    @rate_limited_async_call(GEMINI_RATE_LIMIT)
    async def _get_completion(self, prompt_content: str, system_prompt: str = None, return_cost: bool = False) -> CallResult:
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
                
                if return_cost:
                    # Calculate costs (estimate tokens since Gemini doesn't provide usage info)
                    input_tokens = response.usage_metadata.prompt_token_count
                    output_tokens = response.usage_metadata.candidates_token_count
                    cost = calculate_cost(self.model_name, input_tokens, output_tokens)
                
                    return CallResult(
                        response=response.text.strip(),
                        input_tokens=input_tokens,
                        output_tokens=output_tokens,
                        cost=cost,
                        model_name=self.model_name
                    )
                else:
                    return response.text.strip()
            except Exception as e:
                if attempt < max_retries - 1:
                    print(f"Attempt {attempt + 1} failed for prompt '{prompt_content[:50]}...': {e}")
                    await asyncio.sleep(retry_delay * (2 ** attempt))  # Exponential backoff
                else:
                    print(f"All retries failed for prompt '{prompt_content[:50]}...': {e}")
                    return None
    
    async def batch_invoke(self, prompts: List[str], system_prompt: str = None, return_cost: bool = False) -> List[CallResult]:
        """
        Processes a list of prompts in batches with rate limiting to avoid overwhelming the API.
        
        Args:
            prompts: List of prompts to process
            system_prompt: Optional system prompt
            batch_size: Number of prompts to process in each batch (default: 50)
            delay_between_batches: Delay in seconds between batches (default: 1.0)
        """
        all_results = []
        
        # Process current batch with limited concurrency
        tasks = [self._get_completion(prompt, system_prompt, return_cost) for prompt in prompts]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle exceptions in batch results
        for j, result in enumerate(results):
            if isinstance(result, Exception):
                all_results.append(None)
            else:
                all_results.append(result)
        
        return all_results


class TogetherModel(BaseLLM):
    def __init__(self, model_name: str):
        super().__init__(model_name)
        self.client = Together(api_key=os.environ["TOGETHER_API_KEY"])

    def invoke(self, prompt: str, system_prompt: str = None, return_cost: bool = False) -> CallResult:
        """Generates model output using the TogetherAI API."""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            )
        if return_cost:
        
            # Calculate costs (estimate tokens since Together doesn't provide usage info)
            input_tokens = response.usage.prompt_tokens
            output_tokens = response.usage.completion_tokens
            cost = calculate_cost(self.model_name, input_tokens, output_tokens)
            
            return CallResult(
                response=response.choices[0].message.content,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cost=cost,
                model_name=self.model_name
            )
        else:
            return response.choices[0].message.content
    
    def batch_invoke(self, prompts: List[str], system_prompt: str = None, return_cost: bool = False) -> List[CallResult]:
        responses = list()
        for prompt in prompts:
            response = self.invoke(prompt, system_prompt, return_cost)
            responses.append(response)
        return responses


class ClaudeModel(BaseLLM):
    """Wrapper for Anthropic Claude models."""
    def __init__(self, model_name: str):
        super().__init__(model_name)
        self.client = Anthropic()

    def invoke(self, prompt: str, system_prompt: str = None, return_cost: bool = False) -> CallResult:
        """Generates model output using the Anthropic Messages API."""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        output = self.client.messages.create(
            model=self.model_name,
            messages=messages
        )
        if return_cost:
            # Calculate costs (estimate tokens since Claude doesn't provide usage info in response)
            input_tokens = estimate_tokens(prompt)
            output_tokens = estimate_tokens(output.content[0].text)
            cost = calculate_cost(self.model_name, input_tokens, output_tokens)
            
            return CallResult(
                response=output.content[0].text.strip(),
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cost=cost,
                model_name=self.model_name
            )
        else:
            return output.content[0].text.strip()
    
    def batch_invoke(self, prompts: List[str], system_prompt: str = None, return_cost: bool = False) -> List[CallResult]:
        responses = list()
        for prompt in prompts:
            response = self.invoke(prompt, system_prompt, return_cost)
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
    llm = OpenAIModel(model_name="gpt-4o-mini")
    result = llm.invoke(prompt, return_cost=True)
    print(f"Response: {result.response}")
    print(f"Cost: ${result.cost:.6f}")
    print(f"Input tokens: {result.input_tokens}")
    print(f"Output tokens: {result.output_tokens}")

def test_moderation():
    prompt = "How can I use my political power to abuse others?"
    llm = OpenAIModerationModel(model_name="omni-moderation-latest")
    response = llm.invoke(prompt)
    print(f"Moderation Response: {response}")

async def test_ainvoke():
    """Test function to demonstrate LLM usage"""
    # Test prompt
    prompt = "What are three key benefits of async programming in Python?"
    prompts = [prompt] * 3
    
    # Initialize different OpenAI models
    llm = OpenAIModel(model_name="gpt-4o-mini")
    results = await llm.batch_invoke(prompts, return_cost=True)
    
    total_cost = sum(result.cost for result in results if result is not None)
    total_input_tokens = sum(result.input_tokens for result in results if result is not None)
    total_output_tokens = sum(result.output_tokens for result in results if result is not None)
    
    print(f"Batch processing completed:")
    print(f"Total cost: ${total_cost:.6f}")
    print(f"Total input tokens: {total_input_tokens}")
    print(f"Total output tokens: {total_output_tokens}")

if __name__ == "__main__":
    main()
    asyncio.run(test_ainvoke())
    test_moderation()


