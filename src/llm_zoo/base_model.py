from abc import ABC, abstractmethod

class BaseLLM(ABC):
    """Base class for LLM wrappers"""
    
    def __init__(self, model_name: str, **kwargs):
        self.model_name = model_name
        self.model_kwargs = kwargs

    @abstractmethod
    def invoke(self, prompt: str) -> str:
        """Generate response from the LLM"""
        pass