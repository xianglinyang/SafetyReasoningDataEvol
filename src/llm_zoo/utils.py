import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
import logging

from src.llm_zoo.model_configs import get_system_prompt

logger = logging.getLogger(__name__)


def load_tokenizer(model_name_or_path):
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, padding_side='left')
    tokenizer.padding_side = 'left'
    # Properly set up padding token and ID
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
        else:
            # If no eos token, use a common special token
            tokenizer.pad_token = ' '
            tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(' ')


def load_model(model_name_or_path, device_map="cuda:0", torch_dtype=torch.bfloat16):
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, torch_dtype=torch_dtype, device_map=device_map)
    model.eval()
    return model


def prompt2messages(prompt, model_name_or_path):
    system_prompt = get_system_prompt(model_name_or_path)
    messages = list()
    if system_prompt is not None:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    return messages


def batch_invoke(model, tokenizer, questions, batch_size=8, max_new_tokens=2048):
    """Generate answers for a batch of questions"""
    current_batch_size = batch_size

    all_responses = []
    for i in tqdm(range(0, len(questions), current_batch_size), desc="Generating answers"):
        batch_prompts = questions[i:i + current_batch_size]
        # Clear CUDA cache between batches
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        try:
            
            # Prepare prompts
            all_messages = []

            for prompt in batch_prompts:
                messages = prompt2messages(prompt)
                formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False)
                all_messages.append(formatted_prompt)
        # Tokenize
            inputs = tokenizer(
                all_messages, 
                return_tensors="pt", 
                padding=True,
                truncation=True,
                pad_to_multiple_of=8,
                max_length=max_new_tokens
            ).to(model.device)
                
            # Generate
            with torch.no_grad():
                outputs = model.generate(
                    input_ids=inputs['input_ids'],
                    attention_mask=inputs['attention_mask'],
                    max_new_tokens=max_new_tokens,
                    temperature=0.1,
                    do_sample=True,
                    top_p=0.9,
                    pad_token_id=tokenizer.pad_token_id,
                    use_cache=True
                )
            # Decode
            for j, output in enumerate(outputs):
                # Get the length of the input sequence
                input_length = inputs['input_ids'][j].shape[0]
                # Decode only the generated part (everything after the input)
                decoded_output = tokenizer.decode(output[input_length:], skip_special_tokens=True)
                all_responses.append(decoded_output)
                
        except RuntimeError as e:
            if "out of memory" in str(e) or "device-side assert triggered" in str(e):
                # If we hit OOM, reduce batch size and retry this batch
                torch.cuda.empty_cache()
                current_batch_size = max(1, current_batch_size // 2)
                logger.warning(f"Reduced batch size to {current_batch_size} due to memory error")
                i -= current_batch_size  # Retry this batch
                continue
            else:
                raise e
    return all_responses

if __name__ == "__main__":
    model_name_or_path = "meta-llama/Llama-3.1-8B-Instruct"
    model = load_model(model_name_or_path)
    tokenizer = load_tokenizer(model_name_or_path)
    questions = ["What is the capital of France?", "What is the capital of Germany?"]
    responses = batch_invoke(model, tokenizer, questions, batch_size=2)
    print(responses)