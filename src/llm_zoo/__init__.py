'''
A LLM wrapper for all models
'''
# from src.llm_zoo.api_base_models import OpenAILLM
# from src.llm_zoo.code_base_models import HuggingFaceLLM


# # TODO: add more models
# # __all__ = ['llm_generate', 'OpenAILLM']

# LLM_MODELS = {
#     "OPENAI_LLM": OpenAILLM.CHAT_MODELS | OpenAILLM.INSTRUCT_MODELS,
# }

# async def llm_generate(model_name: str, prompt: str, **kwargs) -> str:
#     if model_name in LLM_MODELS["OPENAI_LLM"]:
#         llm = OpenAILLM(model_name=model_name, **kwargs)
#     else:
#         raise ValueError(f"Unknown model: {model_name}. Must be one of {LLM_MODELS['OPENAI_LLM']}")
#     response = await llm.generate(prompt)
#     return response