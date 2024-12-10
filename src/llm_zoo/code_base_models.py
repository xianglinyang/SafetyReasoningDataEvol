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
'''

import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, LoraConfig, get_peft_model

from src.llm_zoo.base_model import BaseLLM
from src.llm_zoo.model_configs import MODEL_CONFIGS

class HuggingFaceLLM(BaseLLM):
    def __init__(self, model_name_or_path="meta-llama/Llama-2-7b-chat-hf", model_abbr="llama2", torch_dtype=torch.bfloat16, device="cuda"):
        super().__init__(model_name_or_path)
        self.model_name_or_path=model_name_or_path
        self.model_abbr=model_abbr
        self.torch_dtype=torch_dtype
        self.device=device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self._load_model(model_name_or_path, device, torch_dtype)
        
        # -- model configs --
        self.separator = MODEL_CONFIGS[model_abbr]["sep_token"]
        self.user_tag = MODEL_CONFIGS[model_abbr]["user_tag"]
        self.assistant_tag = MODEL_CONFIGS[model_abbr]["assistant_tag"]
        self.system_tag = MODEL_CONFIGS[model_abbr]["system"]
    

# model = load_model(args.model_path, dtype)

# # resize embeddings if needed (e.g. for LlamaTokenizer)
# embedding_size = model.get_input_embeddings().weight.shape[0]
# if len(tokenizer) > embedding_size:
#     model.resize_token_embeddings(len(tokenizer))

# if args.initialize_lora:
#     assert not isinstance(model, PeftModel)
#     lora_config = LoraConfig(
#         task_type=TaskType.CAUSAL_LM,
#         inference_mode=False,
#         r=args.lora_r,
#         lora_alpha=args.lora_alpha,
#         lora_dropout=args.lora_dropout,
#         target_modules=args.lora_target_modules,
#     )
#     model = get_peft_model(model, lora_config)

# if isinstance(model, PeftModel):
#     model.print_trainable_parameters()

    
    def _load_tokenizer(self, model_name_or_path):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({"pad_token": "<pad>"})

    def _load_model(self, model_name_or_path, device, torch_dtype):
        is_peft = os.path.exists(os.path.join(model_name_or_path, "adapter_config.json"))
        if is_peft:
            # load this way to make sure that optimizer states match the model structure
            config = LoraConfig.from_pretrained(model_name_or_path)
            base_model = AutoModelForCausalLM.from_pretrained(
                config.base_model_name_or_path, torch_dtype=torch_dtype, device_map=self.device, ignore_mismatched_sizes=True)
            self.model = PeftModel.from_pretrained(
                base_model, model_name_or_path, device_map=self.device, ignore_mismatched_sizes=True)
            for name, param in self.model.named_parameters():
                if 'lora' in name or 'Lora' in name:
                    param.requires_grad = True
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name_or_path, torch_dtype=torch_dtype, device_map=self.device, ignore_mismatched_sizes=True)

        # resize embeddings if needed (e.g. for LlamaTokenizer)
        embedding_size = self.model.get_input_embeddings().weight.shape[0]
        if len(self.tokenizer) > embedding_size:
            self.model.resize_token_embeddings(len(self.tokenizer))

    def format_prompt(self, prompt, system=True):
        if system:
            one_shot_template = "{system}{user_tag}{instruction}{assistant_tag}{separator}"
            return one_shot_template.format(
                system=system,
                user_tag=self.user_tag, 
                instruction=prompt, 
                assistant_tag=self.assistant_tag,
                separator=self.separator)
        else:
            one_shot_template = "{user_tag}{instruction}{assistant_tag}{separator}"
            return one_shot_template.format(
                user_tag=self.user_tag, assistant_tag=self.assistant_tag,
                instruction=prompt, separator=self.separator)

    def invoke(self, prompt, max_new_tokens=2048, temperature=0.1, system=True, verbose=False):
        # Use model specific format
        prompt = self.format_prompt(prompt, system=system)

        # 3: Tokenize the chat (This can be combined with the previous step using tokenize=True)
        inputs = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
        # Move the tokenized inputs to the same device the model is on (GPU/CPU)
        inputs = {key: tensor.to(self.model.device) for key, tensor in inputs.items()}
        if verbose: print("Tokenized inputs:\n", inputs)
        
        # 4: Generate text from the model
        outputs = self.model.generate(**inputs, max_new_tokens=max_new_tokens, temperature=temperature)
        if verbose: print("Generated tokens:\n", outputs)

        # 5: Decode the output back to a string
        decoded_output = self.tokenizer.decode(outputs[0][inputs['input_ids'].size(1):], skip_special_tokens=True)
        if verbose: print("Decoded output:\n", decoded_output)
        
        return decoded_output


# test function
if __name__ == "__main__":
    model_path = "out/checkpoint-312"
    model = HuggingFaceLLM(model_name_or_path=model_path, model_abbr="llama2", device="cuda")
    prompt = "What is the capital of France?"
    print(model.invoke(prompt, verbose=False))