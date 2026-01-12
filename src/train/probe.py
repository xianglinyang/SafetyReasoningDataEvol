'''
算loss

返回每个样本 answer 前 K token 的 mean NLL（只在 answer token 上算 loss）
'''
from typing import List, Tuple, Dict, Any
import torch
from vllm import LLM, SamplingParams
import numpy as np

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
    llm: LLM,
    tokenizer,
    questions: List[str],
    answers: List[str],
    k_ans_tokens: int = 64,
    chunk_size: int = 2048,   # 一次喂给 vLLM 的样本数（vLLM 会自己再动态 batching）
) -> torch.Tensor:
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

    idx = 0
    while idx < len(prompts):
        batch_prompts = prompts[idx: idx + chunk_size]
        batch_prefix = prefix_lens[idx: idx + chunk_size]

        req_outs = llm.generate(batch_prompts, sp)

        for j, ro in enumerate(req_outs):
            # ro.prompt_logprobs: list，长度=prompt tokens；每个元素通常是 dict{token_id: Logprob} 或 None
            plps = ro.prompt_logprobs
            pre = batch_prefix[j]

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

            out[idx + j] = (s / max(c, 1))

        idx += chunk_size

    return out

def probe_operator_gradient_direction(
    model_name_or_path: str,
    tokenizer, 
    dataset: List[Dict[str, Any]],
    k_ans_tokens: int = 64,
    chunk_size: int = 2048,
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

    
    # Step 2: Calculate losses for all mutations
    llm = LLM(
        model=model_name_or_path,
        dtype="bfloat16",
        gpu_memory_utilization=0.9,
        max_model_len=1024, 
    )

    all_losses = vllm_mean_nll_on_answer_span(
        llm, tokenizer, all_texts, all_answers, k_ans_tokens=k_ans_tokens, chunk_size=chunk_size)

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


if __name__ == "__main__":
    from src.data_utils.RobustSCoT_datasets import data_reader
    from transformers import AutoTokenizer
    import time

    model_name_or_path = "meta-llama/Meta-Llama-3-8B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

    dataset_name = "circuitbreaker_diverse"
    _, dataset = data_reader(dataset_name)

    dataset = dataset[:500]

    t1 = time.time()

    results = probe_operator_gradient_direction(
        model_name_or_path,
        tokenizer,
        dataset,
        k_ans_tokens=64,
        chunk_size=2048,
     )
    t2 = time.time()
    for result in results:
        for strategy, loss in result['probe_results'].items():
            print(f"{strategy}: {loss}")
        print("-"*100)
    
    from src.train.targeted_generation import select_strategies
    new_dataset = select_strategies(results, top_ratio=0.1)

    for data in new_dataset:
        print(data['selected_strategy'])
        print(data['is_mask'])
        print("-"*100)
    
    print(f"Time taken: {t2 - t1} seconds for 500 samples")



# if __name__ == "__main__":
#     from src.data_utils.RobustSCoT_datasets import data_reader
#     from transformers import AutoTokenizer, AutoModelForCausalLM

#     tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
#     model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct", torch_dtype=torch.bfloat16)

#     dataset_name = "circuitbreaker_diverse"
#     _, dataset = data_reader(dataset_name)
#     dataset = dataset[:200]

#     results = probe_operator_gradient_direction(
#         model,
#         tokenizer,
#         dataset,
#         batch_size=32,
#         max_length=1024,
#         device="cuda:0",
#      )
#     for result in results:
#         for strategy, loss in result['probe_results'].items():
#             print(f"{strategy}: {loss}")
#         print("-"*100)
    
#     from src.train.targeted_generation import select_strategies
#     new_dataset = select_strategies(results, top_ratio=0.1)

#     for data in new_dataset:
#         print(data['selected_strategy'])
#         print(data['is_mask'])
#         print("-"*100)
    
    
