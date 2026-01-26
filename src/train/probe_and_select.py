'''
算loss

返回每个样本 answer 前 K token 的 mean NLL（只在 answer token 上算 loss）
'''
from typing import List, Tuple, Dict, Any
import torch
from vllm import LLM, SamplingParams
import numpy as np
import argparse
from transformers import AutoTokenizer
import json
import numpy as np
import random
import os
import asyncio

# ------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------
def build_prompts_and_prefix_lens(
    tokenizer,
    questions: List[str],
    answers: List[str],
    k_ans_tokens: int = 64,
) -> Tuple[List[str], List[int]]:
    """
    返回：
      prompts: prefix + truncated_answer 的字符串 prompt
      prefix_lens: prefix 的 token 长度（用于切出 answer span）
    """
    prompts, prefix_lens = [], []

    for q, a in zip(questions, answers):
        # prefix: user + assistant header（关键：add_generation_prompt=True）
        prefix_text = tokenizer.apply_chat_template(
            [{"role": "user", "content": q}],
            tokenize=False,
            add_generation_prompt=True,
        )

        # truncate answer by tokens (K)
        ans_ids = tokenizer(a, add_special_tokens=False).input_ids[:k_ans_tokens]
        a_trunc = tokenizer.decode(ans_ids, skip_special_tokens=False)

        prompt = prefix_text + a_trunc

        # prefix token length (用同一个 tokenizer 估计)
        pre_ids = tokenizer(prefix_text, add_special_tokens=False).input_ids
        prefix_lens.append(len(pre_ids))
        prompts.append(prompt)

    return prompts, prefix_lens


def vllm_mean_nll_on_answer_span(
    model_name_or_path: str,  # Can be LLM instance or model path string
    tokenizer,
    questions: List[str],
    answers: List[str],
    k_ans_tokens: int = 64,
    gpu_memory_utilization: float = 0.9,
    max_model_len: int = 1024,
    tensor_parallel_size: int = 1,
) -> torch.Tensor:
    # If llm is a string (model path), create LLM instance
    # Otherwise assume it's already an LLM instance
    llm_instance = LLM(
        model=model_name_or_path,
        dtype="bfloat16",
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
        tensor_parallel_size=tensor_parallel_size,  # Use only 1 GPU
    )
    
    prompts, prefix_lens = build_prompts_and_prefix_lens(
        tokenizer, questions, answers, k_ans_tokens=k_ans_tokens
    )

    sp = SamplingParams(
        max_tokens=50,          # 不生成，只算 prompt 的 logprobs
        prompt_logprobs=1,     # 返回每个 prompt token 的 logprob（对应真实 token）
        temperature=0.0,
    )

    out = torch.empty(len(prompts), dtype=torch.float32)

    def _get_logprob(x):
        # x 可能是 Logprob 对象/float/其它，尽量兼容不同 vLLM 版本
        if hasattr(x, "logprob"):
            return float(x.logprob)
        return float(x)

    req_outs = llm_instance.generate(prompts, sp)

    for j, ro in enumerate(req_outs):
        plps = ro.prompt_logprobs
        pre = prefix_lens[j]

        # 只累计 answer span 的 logprob（从 prefix_len 开始到末尾）
        s = 0.0
        c = 0
        for t in range(pre, len(plps)):
            item = plps[t]
            if item is None:
                continue
            # item 常见是 dict: {token_id: Logprob}
            if isinstance(item, dict) and len(item) > 0:
                v = next(iter(item.values()))
                lp = _get_logprob(v)
            else:
                # 有些版本可能直接给 float / list
                try:
                    lp = _get_logprob(item)
                except Exception:
                    continue
            s += (-lp)
            c += 1

        out[j] = (s / max(c, 1))

    return out

def probe_operator_gradient_direction(
    model_name_or_path: str,
    tokenizer, 
    dataset: List[Dict[str, Any]],
    k_ans_tokens: int = 64,
    gpu_memory_utilization: float = 0.9,
    max_model_len: int = 1024,
    tensor_parallel_size: int = 1,
):
    """
    Given a dataset, calculate the direction of steepest loss ascent for each mutation strategy. Each strategy might have multiple mutations.
    Args:
        target_model: The model to probe.
        tokenizer: The tokenizer to use.
        dataset: The dataset to probe. The structure of the dataset is:
            [
                {
                    "question": str,
                    "answer": str,
                    "mutations": List[
                        {
                            "strategy": str,
                            "text": str
                        },
                        ...
                    ]
                    {
                        "strategy": str,
                        "text": str
                    },
                    ...
                },
                ...
            ]
        batch_size: Batch size for processing.
        max_length: Maximum sequence length.
        device: Device to use for computation.
        use_amp: Whether to use automatic mixed precision.
    Returns:
        List of dicts, each dict contains:
            - "question": str,
            - "answer": str
            - "probe_results": Dict[str, float]
            {
            "strategy": avg(loss_i) - loss_original,
            }
    """
    # Step 1: Calculate losses for all (question, answer) pairs
    original_questions = [data['question'] for data in dataset]
    original_answers = [data['answer'] for data in dataset]
    
    mutation_texts = []
    mutation_answers = []
    mutation_metadata = []  # Store (data_idx, strategy) for each mutation
    
    for data_idx, data in enumerate(dataset):
        answer = data['answer']
        mutations = data.get('mutations', [])
        
        for mutation in mutations:
            mutation_texts.append(mutation['text'])
            mutation_answers.append(answer)
            mutation_metadata.append({
                'data_idx': data_idx,
                'strategy': mutation['strategy']
            })
    
    all_texts = original_questions + mutation_texts
    all_answers = original_answers + mutation_answers

    all_losses = vllm_mean_nll_on_answer_span(
        model_name_or_path, tokenizer, all_texts, all_answers, k_ans_tokens=k_ans_tokens, gpu_memory_utilization=gpu_memory_utilization, max_model_len=max_model_len, tensor_parallel_size=tensor_parallel_size)

    original_losses = all_losses[:len(original_questions)]
    mutation_losses  = all_losses[len(original_questions):]

    # Step 3: Aggregate results by strategy for each data point
    results = []
    for data_idx, data in enumerate(dataset):
        original_loss = original_losses[data_idx].item()
        
        # Collect all mutations for this data point, grouped by strategy
        strategy_diffs = {}  # strategy -> list of loss differences
        
        for mut_idx, metadata in enumerate(mutation_metadata):
            if metadata['data_idx'] == data_idx:
                strategy = metadata['strategy']
                mut_loss = mutation_losses[mut_idx].item()
                loss_diff = mut_loss - original_loss
                
                if strategy not in strategy_diffs:
                    strategy_diffs[strategy] = []
                strategy_diffs[strategy].append(loss_diff)
        
        # Average the loss differences for each strategy
        strategy_avg_diffs = {
            strategy: float(np.mean(diffs))
            for strategy, diffs in strategy_diffs.items()
        }
        
        results.append({
            'question': data['question'],
            'answer': data['answer'],
            'original_loss': original_loss,
            'probe_results': strategy_avg_diffs
        })
    
    return results


def select_strategies(harmful_dataset_with_losses, top_ratio=0.1):
    """
    Select the strategies with the highest loss differences.
    """
    # 1. select the strategies with the highest loss differences
    losses = []
    strategies = []
    for data in harmful_dataset_with_losses:
        # choose max
        prob = random.random()
        if prob < 0.1:
            # randomly choose a strategy
            strategy = random.choice(list(data['probe_results'].keys()))
            loss_diff = data['probe_results'][strategy]
            losses.append(loss_diff)
            strategies.append(strategy)
        else:
            strategy, loss_diff = max(data['probe_results'].items(), key=lambda x: x[1])
            losses.append(loss_diff)
            strategies.append(strategy)

    # 2. mask list with topK
    losses_np = np.array(losses)
    topK_idxs = np.argsort(losses_np)[::-1][:int(len(losses_np)*top_ratio)]
    # turn it into a mask
    mask = [1 for _ in range(len(losses_np))]
    for idx in topK_idxs:
        mask[idx] = 0

    # 3. write them back to the dataset
    for data, strategy, loss_diff, idx in zip(harmful_dataset_with_losses, strategies, losses, mask):
        data['selected_strategy'] = strategy
        data['loss_diff'] = loss_diff
        data['is_mask'] = idx
    return harmful_dataset_with_losses

def load_probe_results(probe_results_path: str) -> List[Dict[str, Any]]:
    with open(probe_results_path, 'r') as f:
        return json.load(f)

def load_processed_dataset(processed_dataset_path: str) -> List[Dict[str, Any]]:
    benign_dataset_path = os.path.join(processed_dataset_path, 'benign_dataset.json')
    harmful_dataset_path = os.path.join(processed_dataset_path, 'harmful_dataset_with_losses.json')
    with open(benign_dataset_path, 'r') as f:
        benign_dataset = json.load(f)
    with open(harmful_dataset_path, 'r') as f:
        harmful_dataset = json.load(f)
    return benign_dataset, harmful_dataset

# ------------------------------------------------------------
# Main function
# ------------------------------------------------------------
# do probe and select and generate mutations and save the dataset to json for later training

from src.data_utils.RobustSCoT_datasets import data_reader
from src.llm_zoo import load_model
from src.evol.question_evol import QuestionEvol

async def main():
    
    parser = argparse.ArgumentParser(description="Probe operator gradient direction using vLLM")
    # probe arguments
    parser.add_argument("--model_name_or_path", type=str, required=True,
                        help="Path to the model or HuggingFace model name")
    parser.add_argument("--dataset_name", type=str, required=True,
                        help="Dataset name to load")
    parser.add_argument("--k_ans_tokens", type=int, default=64,
                        help="Number of answer tokens to consider")

    # select arguments
    parser.add_argument("--top_ratio", type=float, default=0.1,
                        help="Top ratio of strategies to select")
    parser.add_argument("--mutation_top_ratio", type=float, default=0.1,
                        help="Top ratio of strategies to select")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Maximum number of samples to process (for testing)")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9,
                        help="GPU memory utilization for vLLM")
    parser.add_argument("--max_model_len", type=int, default=1024,
                        help="Maximum model length for vLLM")
    parser.add_argument("--tensor_parallel_size", type=int, default=1,
                        help="Number of GPUs to use for tensor parallelism")
    # mutation arguments
    parser.add_argument("--mutation_llm", type=str, default="openai/gpt-4.1",
                        help="LLM to use for the mutations")
    parser.add_argument("--mutation_num", type=int, default=5,
                        help="Number of mutations to generate")
    parser.add_argument("--mutation_alpha", type=float, default=0.9,
                        help="Alpha for the mutations")
    parser.add_argument("--mutation_demo_selected_strategy", type=str, default="diverse",
                        help="The strategy to select the demo examples")

    # save arguments
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save probe results (JSON)")
    parser.add_argument("--epoch", type=int, default=0,
                        help="Epoch number")
    
    args = parser.parse_args()

    # 0. load dataset
    benign_dataset, harmful_dataset = data_reader(args.dataset_name)
    # benign_dataset = benign_dataset[:50]
    # harmful_dataset = harmful_dataset[:50]

    # 1. probe
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    harmful_dataset_with_losses = probe_operator_gradient_direction(
        args.model_name_or_path, 
        tokenizer, 
        harmful_dataset,
        k_ans_tokens=args.k_ans_tokens,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        tensor_parallel_size=args.tensor_parallel_size
    )

    # 2. select strategies and hard samples
    harmful_dataset_with_losses = select_strategies(harmful_dataset_with_losses, top_ratio=args.mutation_top_ratio)

    # 3. generate mutations, select the ones without mask
    questions = [data['question'] for data in harmful_dataset_with_losses if not data['is_mask']]
    answers = [data['answer'] for data in harmful_dataset_with_losses if not data['is_mask']]
    targeted_strategies = [data['selected_strategy'] for data in harmful_dataset_with_losses if not data['is_mask']]
        
    question_evol = QuestionEvol()
    llm_client = load_model(args.mutation_llm)
    mutation_num = args.mutation_num
    alpha = args.mutation_alpha
    demo_selected_strategy = args.mutation_demo_selected_strategy
    instruction_variants = await question_evol.generate_prompt_variants_batch_targeted(questions, llm_client, targeted_strategies, num=mutation_num, alpha=alpha, demo_selected_strategy=demo_selected_strategy)
    
    mutation_candidates = dict()
    for question, mutations in instruction_variants.items():
        mutation_candidates[question] = [mutation['text'] for mutation in mutations]

    # 4. compute the losses of the mutations and select the mutations with the highest losses
    
    mutation_questions = []
    mutation_answers = []
    for (question, mutations), answer in zip(mutation_candidates.items(), answers):
        mutation_questions.extend(mutations)
        mutation_answers.extend([answer] * len(mutations))
    
    new_losses = vllm_mean_nll_on_answer_span(
        model_name_or_path=args.model_name_or_path, 
        tokenizer=tokenizer, 
        questions=mutation_questions, 
        answers=mutation_answers, 
        k_ans_tokens=args.k_ans_tokens, 
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        tensor_parallel_size=args.tensor_parallel_size
    )

    new_losses_list = new_losses.cpu().numpy().tolist()
    
    losses_map = dict()
    for mutation_question, loss in zip(mutation_questions, new_losses_list):
        losses_map[mutation_question] = loss
    
    # Clean up large lists to free memory
    del mutation_questions, mutation_answers, new_losses_list
    del new_losses
    
    for data in harmful_dataset_with_losses:
        if not data['is_mask']:
            question = data['question']
            mutations = mutation_candidates.get(question, [])
            if len(mutations) == 0:
                # Skip if no mutations were generated
                print(f"Warning: No mutations generated for question: {question[:50]}...")
                continue
            losses = [losses_map[mutation] for mutation in mutations]
            data['candidates'] = mutations
            data['losses'] = losses
            data['selected_mutation'] = mutations[np.argmax(losses)]
    
    # Clean up mutation_candidates and losses_map to free memory
    del mutation_candidates, losses_map
            
    # 5. prepare for training by splitting into train and val
    # Add data_type field to datasets before shuffling
    for data in benign_dataset:
        data['data_type'] = 'benign'
    
    # Mark harmful data
    for data in harmful_dataset_with_losses:
        data['data_type'] = 'harmful'

    save_dir = os.path.join(args.output_dir, f"epoch_{args.epoch}")
    os.makedirs(save_dir, exist_ok=True)
    
    harmful_save_path = os.path.join(save_dir, f"harmful_dataset_with_losses.json")
    with open(harmful_save_path, 'w') as f:
        json.dump(harmful_dataset_with_losses, f)
    
    benign_save_path = os.path.join(save_dir, f"benign_dataset.json")
    with open(benign_save_path, 'w') as f:
        json.dump(benign_dataset, f)


if __name__ == "__main__":
    asyncio.run(main())

    
    
