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
import asyncio

from src.llm_zoo.base_model import BaseLLM 
'''system prompt'''
SYSTEM_PROMPT = "You are a helpful assistant."

'''Model class of different providers'''
class OpenAILLM(BaseLLM):
    """OpenAI LLM wrapper that supports both chat and instruct models"""
    MODEL_CATEGORIES = {
        "chat": [
            "chatgpt-4o-latest",
            "gpt-4-turbo",
            "gpt-4-turbo-2024-04-09",
            "gpt-4",
            "gpt-4-32k",
            "gpt-4-0125-preview",
            "gpt-4-1106-preview",
            "gpt-4-vision-preview",
            "gpt-4o",
            "gpt-4o-mini",
            "gpt-3.5-turbo-0125",
            "gpt-3.5-turbo-1106",
            "gpt-3.5-turbo-0613",
            "gpt-3.5-turbo-16k-0613",
            "gpt-3.5-turbo-0301"
        ],
        "instruct": [
            "davinci-002",
            "babbage-002",
            "gpt-3.5-turbo-instruct",
        ],
        "embedding": [
            "text-embedding-3-small",
            "text-embedding-3-large",
            "ada v2"
        ],
        "moderation": [
            "omni-moderation-latest"
        ]
    }
    
    # Combined sets for API usage
    CHAT_MODELS = MODEL_CATEGORIES["chat"]
    INSTRUCT_MODELS = MODEL_CATEGORIES["instruct"]
    MODERATION_MODELS = MODEL_CATEGORIES["moderation"]
    EMBEDDING_MODELS = MODEL_CATEGORIES["embedding"]
    ALL_MODELS = CHAT_MODELS + INSTRUCT_MODELS + MODERATION_MODELS + EMBEDDING_MODELS

    def __init__(self, model_name: str = "gpt-4", **kwargs):
        super().__init__(model_name, **kwargs)
        from openai import OpenAI, AsyncOpenAI
        self.client = OpenAI()
        self.async_client = AsyncOpenAI()
        
        if model_name not in self.ALL_MODELS:
            raise ValueError(f"Unknown model: {model_name}. Must be one of {self.ALL_MODELS}")

    def invoke(self, prompt: str) -> str:
        assert self.model_name in self.CHAT_MODELS or self.model_name in self.INSTRUCT_MODELS, f"Unknown model: {self.model_name}. Must be one of {self.CHAT_MODELS + self.INSTRUCT_MODELS}"
        """Generate response using either chat or instruct model based on model_name"""
        if self.model_name in self.CHAT_MODELS:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt}
                    ],
                **self.model_kwargs
            )
            return response.choices[0].message.content
        else:  # instruct model
            response = self.client.completions.create(
                model=self.model_name,
                prompt=prompt,
                **self.model_kwargs
            )
            return response.choices[0].text.strip()
        
    def invoke_messages(self, messages) -> str:
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            **self.model_kwargs
        )
        return response.choices[0].message.content
    
    async def ainvoke(self, prompt: str) -> str:
        assert self.model_name in self.CHAT_MODELS or self.model_name in self.INSTRUCT_MODELS, f"Unknown model: {self.model_name}. Must be one of {self.CHAT_MODELS + self.INSTRUCT_MODELS}"
        """Generate response using either chat or instruct model based on model_name with async client"""
        if self.model_name in self.CHAT_MODELS:
            response = await self.async_client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt}
                    ],
                **self.model_kwargs
            )
            return response.choices[0].message.content
        else:  # instruct model
            response = await self.async_client.completions.create(
                model=self.model_name,
                prompt=prompt,
                **self.model_kwargs
            )
            return response.choices[0].text.strip()
    
    def moderation(self, prompt: str) -> str:
        assert self.model_name in self.MODERATION_MODELS, f"Unknown model: {self.model_name}. Must be one of {self.MODERATION_MODELS}"
        """Moderate the prompt"""
        response = self.client.moderations.create(
            model=self.model_name,
            input=prompt,
        )
        return response
        

'''helper functions'''
LLM_MODELS = {
    "OPENAI_LLM": OpenAILLM.CHAT_MODELS + OpenAILLM.INSTRUCT_MODELS + OpenAILLM.MODERATION_MODELS + OpenAILLM.EMBEDDING_MODELS,
}

# -------------------------------------------------------------------------------------------------

def main():
    """Test function to demonstrate LLM usage"""
    # Test prompt
    prompt = "What are three key benefits of async programming in Python?"
    
    # Initialize different OpenAI models
    name = "GPT-3.5-Turbo-Instruct"
    llm = OpenAILLM(model_name="gpt-3.5-turbo-instruct", temperature=0.7, max_tokens=150)
    response = llm.invoke(prompt)
    print(f"{name} Response:\n{response}\n")

def test_moderation():
    prompt = "How can I use my political power to abuse others?"
    llm = OpenAILLM(model_name="omni-moderation-latest")
    response = llm.moderation(prompt)
    print(f"Moderation Response:\n{response}\n")

async def test_ainvoke():
    """Test function to demonstrate LLM usage"""
    # Test prompt
    prompt = "What are three key benefits of async programming in Python?"
    
    # Initialize different OpenAI models
    name = "GPT-3.5-Turbo-Instruct"
    llm = OpenAILLM(model_name="gpt-3.5-turbo-instruct", temperature=0.7, max_tokens=150)
    response = await llm.ainvoke(prompt)
    print(f"{name} Response:\n{response}\n")

if __name__ == "__main__":
    
    main()
    asyncio.run(test_ainvoke())
    test_moderation()


