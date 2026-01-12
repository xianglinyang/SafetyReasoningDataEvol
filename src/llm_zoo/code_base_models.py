'''
A LLM wrapper for all models from huggingface or a local model

Available models:
Chat is aimed at conversations, questions and answers, back and forth
Instruct is for following an instruction to complete a task.

Instruct Following:
meta-llama/Llama-3.2-1B-Instruct
meta-llama/Llama-3.2-3B-Instruct
meta-llama/Meta-Llama-3-8B-Instruct
meta-llama/Llama-2-7b-chat-hf
meta-llama/Llama-2-13b-hf

mistralai/Mistral-7B-Instruct-v0.1

Qwen/Qwen2.5-7B-Instruct

count the latency and token usage as well
'''

import os
import asyncio
import time
import torch
import logging
from typing import List
from transformers import AutoTokenizer, AutoModelForCausalLM
from vllm import LLM, SamplingParams

from src.utils.logging_utils import setup_logging
from src.llm_zoo.base_model import BaseLLM
from src.llm_zoo.model_configs import get_formatted_prompt, get_stop_tokens

logger = logging.getLogger(__name__)

# HuggingFace models
class HuggingFaceModel(BaseLLM):
    def __init__(self, model_name_or_path, torch_dtype=torch.bfloat16, device="cuda", **kwargs):
        super().__init__(model_name_or_path, **kwargs)
        self.model_name_or_path=model_name_or_path
        self.torch_dtype=torch_dtype
        self.device=device
        self._load_tokenizer()
        self._load_model()

    def _load_tokenizer(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({"pad_token": "<pad>"})
        # Set padding side to left for decoder-only models
            # Setting `pad_token_id` to `eos_token_id`:2 for open-end generation.
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.tokenizer.padding_side = "left"

    def _load_model(self):
        self.model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=self.model_name_or_path, torch_dtype=self.torch_dtype, device_map=self.device)
        logger.info("model loaded")

    def invoke(self, prompt, system_prompt: str = None, verbose=False):
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False)
        
        # 3: Tokenize the chat (This can be combined with the previous step using tokenize=True)
        inputs = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
        # Move the tokenized inputs to the same device the model is on (GPU/CPU)
        inputs = {key: tensor.to(self.model.device) for key, tensor in inputs.items()}
        if verbose: logger.info(f"Tokenized inputs:\n{inputs}")
        
        # 4: Generate text from the model
        outputs = self.model.generate(**inputs, do_sample=True)
        if verbose: logger.info(f"Generated tokens:\n{outputs}")

        # 5: Decode the output back to a string
        decoded_output = self.tokenizer.decode(outputs[0][inputs['input_ids'].size(1):], skip_special_tokens=True)
        if verbose: logger.info(f"Decoded output:\n{decoded_output}")
        
        return decoded_output
    
    def batch_invoke(self, prompts: List[str], system_prompt: str = None, verbose=False):
        responses = list()
        for prompt in prompts:
            response = self.invoke(prompt, system_prompt, verbose)
            responses.append(response)
        return responses

# vllm models
class VLLMModel(BaseLLM):
    def __init__(self, model_name_or_path: str, device: str = "cuda", torch_dtype: torch.dtype = torch.bfloat16, tensor_parallel_size: int = 1, gpu_memory_utilization: float = 0.95, max_model_len: int = 8192):
        '''
        Args:
            model_name: str, the name of the model
            device: str, the device to use
            torch_dtype: torch.dtype, the dtype to use
            tensor_parallel_size: int, the number of GPUs to use
            gpu_memory_utilization: float, the GPU memory utilization
            max_model_len: int, maximum model context length (default: None, use model's default)
        '''
        super().__init__(model_name_or_path)
        self.device = device
        self.torch_dtype = torch_dtype
        self.tensor_parallel_size = tensor_parallel_size
        self.gpu_memory_utilization = gpu_memory_utilization
        self.max_model_len = max_model_len
        
        logger.info(f"Initializing LLM with model: {model_name_or_path}...")
        time_start = time.time()
        
        # Build LLM kwargs
        llm_kwargs = {
            "model": model_name_or_path,
            "tensor_parallel_size": tensor_parallel_size,
            "gpu_memory_utilization": gpu_memory_utilization,
            "trust_remote_code": True,
            "dtype": self.torch_dtype,
        }
        
        # Add max_model_len if specified
        if max_model_len is not None:
            llm_kwargs["max_model_len"] = max_model_len
            logger.info(f"Setting max_model_len to {max_model_len}")
        
        self.llm = LLM(**llm_kwargs)
        time_end = time.time()
        logger.info(f"LLM initialization took {time_end - time_start:.2f} seconds.")
    
    def get_latency(self, outputs):
        total_latency_list = []
        prompt_eval_latency_list = []
        first_token_latency_list = []
        sample_latency_list = []
        tokens_per_millisecond_list = []
        
        for i, output in enumerate(outputs):
            # Debug: Check if metrics exist
            if not hasattr(output, 'metrics') or output.metrics is None:
                logger.warning(f"Output {i} has no metrics available")
                continue
                
            try:
                # Access vLLM's built-in metrics (in seconds, convert to milliseconds)
                total_latency_list.append(output.metrics.total_latency * 1000)
                prompt_eval_latency_list.append(output.metrics.prompt_eval_latency * 1000)
                first_token_latency_list.append(output.metrics.first_token_latency * 1000)
                sample_latency_list.append(output.metrics.sample_latency * 1000)
                
                num_generated_tokens = len(output.outputs[0].token_ids)
                if output.metrics.sample_latency > 0:
                    tokens_per_second = num_generated_tokens / output.metrics.sample_latency
                    tokens_per_millisecond_list.append(tokens_per_second)
                    
            except AttributeError as e:
                logger.warning(f"Could not access metrics for output {i}: {e}")
                continue
                
        logger.info(f"Processed {len(total_latency_list)} outputs with metrics out of {len(outputs)} total outputs")
        
        return {
            "avg_total_latency": sum(total_latency_list) / len(total_latency_list) if total_latency_list else 0,
            "avg_prompt_eval_latency": sum(prompt_eval_latency_list) / len(prompt_eval_latency_list) if prompt_eval_latency_list else 0,
            "avg_first_token_latency": sum(first_token_latency_list) / len(first_token_latency_list) if first_token_latency_list else 0,
            "avg_sample_latency": sum(sample_latency_list) / len(sample_latency_list) if sample_latency_list else 0,
            "avg_tokens_per_millisecond": sum(tokens_per_millisecond_list) / len(tokens_per_millisecond_list) if tokens_per_millisecond_list else 0,
            "num_outputs_with_metrics": len(total_latency_list),
            "total_outputs": len(outputs),
        }
        
    def invoke(self, prompt: str, n: int = 1, temperature: float = 0.7, top_p: float = 0.95, max_new_tokens: int = 1024, return_latency: bool = False) -> str:
        stop_tokens = get_stop_tokens(self.model_name)
        sampling_params = SamplingParams(
            n=n,  # Number of output sequences to return for each prompt
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_new_tokens,  # Maximum number of tokens to generate per output. Adjust as needed.
            stop=stop_tokens, # Sequences at which to stop generation.
            truncate_prompt_tokens=8192,  # Truncate prompts that exceed model's max length
        )
        logger.info(f"Using sampling parameters: {sampling_params}")

        logger.info("\nGenerating responses...")

        # vLLM can process a list of prompts in a batch very efficiently.
        # The `llm.generate` method takes a list of prompts and sampling parameters.
        formatted_prompt = get_formatted_prompt(self.model_name, prompt)
        prompts_dataset = [formatted_prompt]
        outputs = self.llm.generate(prompts_dataset, sampling_params)

        latency_metrics = self.get_latency(outputs)
        logger.info(f"Latency metrics: {latency_metrics}")

        if return_latency:
            return outputs[0].outputs[0].text.strip(), latency_metrics
        return outputs[0].outputs[0].text.strip()
    
    def batch_invoke(self, prompts: List[str], n: int = 1, temperature: float = 0.7, top_p: float = 0.95, max_new_tokens: int = 1024, return_latency: bool = False) -> str:
        stop_tokens = get_stop_tokens(self.model_name)
        # Get max_model_len from the LLM config
        max_model_len = getattr(self.llm.llm_engine.model_config, 'max_model_len', 8192)
        
        # Calculate safe truncation length: leave room for output tokens
        safe_truncate_len = max_model_len - max_new_tokens - 10  # -10 as safety buffer
        
        sampling_params = SamplingParams(
            n=n,  # Number of output sequences to return for each prompt
            max_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            stop=stop_tokens,
            truncate_prompt_tokens=safe_truncate_len,  # Truncate to leave room for output
        )
        logger.info(f"Using sampling parameters: {sampling_params}")
        logger.info(f"Max model length: {max_model_len}, Safe truncate length: {safe_truncate_len}")

        logger.info("\nGenerating responses...")
        # vLLM can process a list of prompts in a batch very efficiently.
        formatted_prompts = [get_formatted_prompt(self.model_name, prompt) for prompt in prompts]

        # Manual timing as fallback
        start_time = time.time()
        outputs = self.llm.generate(formatted_prompts, sampling_params)
        end_time = time.time()
        manual_total_time = (end_time - start_time) * 1000  # Convert to milliseconds

        latency_metrics = self.get_latency(outputs)
        latency_metrics['manual_all_latency'] = manual_total_time
        latency_metrics['manual_avg_total_latency'] = manual_total_time / len(prompts)
        
        logger.info(f"Avg total latency: {latency_metrics['avg_total_latency']:.2f} milliseconds")
        logger.info(f"Avg prompt eval latency: {latency_metrics['avg_prompt_eval_latency']:.2f} milliseconds")
        logger.info(f"Avg first token latency: {latency_metrics['avg_first_token_latency']:.2f} milliseconds")
        logger.info(f"Avg sample latency: {latency_metrics['avg_sample_latency']:.2f} milliseconds")
        logger.info(f"Avg tokens per millisecond: {latency_metrics['avg_tokens_per_millisecond']:.2f}")
        logger.info(f"Manual all latency: {latency_metrics['manual_all_latency']:.2f} milliseconds")
        logger.info(f"Manual avg total latency: {latency_metrics['manual_avg_total_latency']:.2f} milliseconds")
        
        # Process and display results
        results = [output.outputs[0].text.strip() for output in outputs]
        logger.info("\nProcessing complete.")

        if return_latency:
            return results, latency_metrics
        return results


def huggingface_test():
    model_path = "meta-llama/Llama-3.1-8B-Instruct"
    model = HuggingFaceModel(model_name_or_path=model_path, device="cuda:0", torch_dtype=torch.bfloat16)
    prompt = "What is the capital of France?"
    print(model.invoke(prompt))

def vllm_test():
    model_name = "meta-llama/Llama-3.1-8B-Instruct" 
    model = VLLMModel(model_name, device="cuda", tensor_parallel_size=1, torch_dtype=torch.bfloat16)
    prompts = ["Solve the following problem:\n\nJen got 3 fish.  They each need $1 worth of food a day.  How much does she spend on food in the month of May?"]*3
    responses = model.batch_invoke(prompts)
    print(responses)


# test function
if __name__ == "__main__":
    # huggingface_test()
    setup_logging(task_name="test", log_level=logging.INFO, log_dir="logs")
    vllm_test()
