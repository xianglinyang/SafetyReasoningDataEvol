import operator
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from typing import List, Union, Optional, Dict, Any
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

def build_batch_inputs_chat(
    tokenizer,
    texts: List[str],
    answers: List[str],
    max_length: int = 2048,
    pad_to_multiple_of: Optional[int] = 8,   # TensorCore/对齐更快
) -> Dict[str, torch.Tensor]:
    assert len(texts) == len(answers)

    # 1) chat template -> full & prefix (batch)
    full_texts, prefix_texts = [], []
    has_template = hasattr(tokenizer, "apply_chat_template")

    for q, a in zip(texts, answers):
        msgs_full = [{"role": "user", "content": q},
                     {"role": "assistant", "content": a}]
        msgs_prefix = [{"role": "user", "content": q}]
        if has_template:
            full_texts.append(tokenizer.apply_chat_template(msgs_full, tokenize=False))
            prefix_texts.append(tokenizer.apply_chat_template(msgs_prefix, tokenize=False))
        else:
            full_texts.append(f"User: {q}\nAssistant: {a}")
            prefix_texts.append(f"User: {q}\nAssistant:")

    # 2) tokenizer batch
    # pad_token 处理
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("Tokenizer has no pad_token_id and no eos_token_id.")
        tokenizer.pad_token = tokenizer.eos_token

    enc_full = tokenizer(
        full_texts,
        return_tensors="pt",
        padding=True,                 # pad 到 batch 内最长（快）
        truncation=True,
        max_length=max_length,
        pad_to_multiple_of=pad_to_multiple_of,
    )
    enc_prefix = tokenizer(
        prefix_texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
        pad_to_multiple_of=pad_to_multiple_of,
    )

    input_ids = enc_full["input_ids"]              # (B, T)
    attention_mask = enc_full["attention_mask"]    # (B, T)

    # prefix_len：用 attention_mask.sum(1)
    prefix_lens = enc_prefix["attention_mask"].sum(dim=1)  # (B,)

    # 3) labels：mask prefix + pad
    labels = input_ids.clone()
    B, T = labels.shape
    pos = torch.arange(T).unsqueeze(0)  # (1, T)
    labels = labels.masked_fill(pos < prefix_lens.unsqueeze(1), -100)
    labels = labels.masked_fill(attention_mask == 0, -100)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


def calculate_losses_from_batch_tensors(
    target_llm,
    batch: Dict[str, torch.Tensor],
    device: Union[str, torch.device],
    use_amp: bool = True,
) -> torch.Tensor:
    target_llm.eval()
    device = torch.device(device)
    target_llm.to(device)

    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    labels = batch["labels"].to(device)

    amp_ok = (use_amp and device.type == "cuda")
    with torch.inference_mode():
        if amp_ok:
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = target_llm(input_ids=input_ids, attention_mask=attention_mask).logits
        else:
            logits = target_llm(input_ids=input_ids, attention_mask=attention_mask).logits

        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        B, Tm1, V = shift_logits.shape

        token_ce = F.cross_entropy(
            shift_logits.view(-1, V),
            shift_labels.view(-1),
            reduction="none",
            ignore_index=-100,
        ).view(B, Tm1)

        valid = (shift_labels != -100).float()
        per_ex_sum = (token_ce * valid).sum(dim=1)
        per_ex_cnt = valid.sum(dim=1).clamp(min=1.0)
        losses = per_ex_sum / per_ex_cnt

    return losses.detach().cpu()


class BatchInputsDataset(Dataset):
    def __init__(self, texts, answers):
        self.texts = texts
        self.answers = answers
    def __len__(self):
        return len(self.texts)
    def __getitem__(self, idx):
        return {"text": self.texts[idx], "answer": self.answers[idx]}


def collate_fn(batch_list, tokenizer, max_length=2048):
    texts = [b["text"] for b in batch_list]
    answers = [b["answer"] for b in batch_list]
    return build_batch_inputs_chat(tokenizer, texts, answers, max_length=max_length, pad_to_multiple_of=8)


def calculate_losses_with_dataloader(model, tokenizer, texts, answers, device="cuda:0", batch_size=10, max_length=2048):
    ds = BatchInputsDataset(texts, answers)
    dl = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        collate_fn=lambda b: collate_fn(b, tokenizer, max_length=max_length),
    )

    out = []
    for batch in tqdm(dl, desc="Calculating losses"):
        out.append(calculate_losses_from_batch_tensors(model, batch, device=device, use_amp=True))
    
    return torch.cat(out, dim=0)

class PerExampleLossWrapper(nn.Module):
    """在每张 GPU 上直接返回 per-example mean NLL（只在 answer token 上）"""
    def __init__(self, model: nn.Module, use_amp: bool = True):
        super().__init__()
        self.model = model
        self.use_amp = use_amp

    def forward(self, input_ids, attention_mask, labels):
        amp_ok = self.use_amp and input_ids.is_cuda
        if amp_ok:
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = self.model(input_ids=input_ids, attention_mask=attention_mask).logits
        else:
            logits = self.model(input_ids=input_ids, attention_mask=attention_mask).logits

        # shift
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        B, Tm1, V = shift_logits.shape

        token_ce = F.cross_entropy(
            shift_logits.view(-1, V),
            shift_labels.view(-1),
            reduction="none",
            ignore_index=-100,
        ).view(B, Tm1)

        valid = (shift_labels != -100).float()
        per_ex_sum = (token_ce * valid).sum(dim=1)
        per_ex_cnt = valid.sum(dim=1).clamp(min=1.0)
        return per_ex_sum / per_ex_cnt  # (local_batch,)


def iter_batches(texts: List[str], answers: List[str], batch_size: int):
    assert len(texts) == len(answers)
    for i in range(0, len(texts), batch_size):
        yield i, texts[i:i+batch_size], answers[i:i+batch_size]


def calculate_losses_multi_gpu(
    target_llm,
    tokenizer,
    texts: List[str],
    answers: List[str],
    batch_size: int = 64,
    max_length: int = 2048,
    device_ids: Optional[List[int]] = [0,1],   # e.g. [0,1,2,3]
    use_amp: bool = True,
):
    assert torch.cuda.is_available(), "CUDA not available"
    if device_ids is None:
        device_ids = list(range(torch.cuda.device_count()))
    assert len(device_ids) >= 1

    # wrap
    base_device = torch.device(f"cuda:{device_ids[0]}")
    wrapper = PerExampleLossWrapper(target_llm, use_amp=use_amp).to(base_device)
    if len(device_ids) > 1:
        wrapper = nn.DataParallel(wrapper, device_ids=device_ids)  # 自动按 batch 维切分

    wrapper.eval()

    # 预分配输出，保证最后顺序和输入一致
    out_losses = torch.empty(len(texts), dtype=torch.float32)

    with torch.inference_mode():
        for start_idx, t_batch, a_batch in tqdm(iter_batches(texts, answers, batch_size), desc="Calculating losses"):
            batch = build_batch_inputs_chat(tokenizer, t_batch, a_batch, max_length=max_length)
            # DataParallel 要求输入在主卡上，它会自动 scatter 到其他卡
            batch = {k: v.to(base_device, non_blocking=True) for k, v in batch.items()}
            losses = wrapper(**batch)  # (B,) on cuda:0
            out_losses[start_idx:start_idx+len(t_batch)] = losses.detach().float().cpu()

    return out_losses  # (N,)


def probe_operator_gradient_direction(
    target_model, 
    tokenizer, 
    dataset: List[Dict[str, Any]],
    batch_size: int = 10,
    max_length: int = 2048,
    device: str = "cuda:0",
    use_amp: bool = True
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
    # Step 1: Calculate original losses for all (question, answer) pairs
    original_questions = [data['question'] for data in dataset]
    original_answers = [data['answer'] for data in dataset]
    
    print("Calculating original losses...")
    original_losses = calculate_losses_with_dataloader(
        target_model, 
        tokenizer, 
        original_questions, 
        original_answers, 
        device=device,
        batch_size=batch_size,
        max_length=max_length
    )
    
    # Step 2: Calculate losses for all mutations
    # Build a flat list of all mutations with their indices
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
    
    # Calculate losses for all mutations at once
    print(f"Calculating losses for {len(mutation_texts)} mutations...")
    if len(mutation_texts) > 0:
        mutation_losses = calculate_losses_with_dataloader(
            target_model,
            tokenizer,
            mutation_texts,
            mutation_answers,
            device=device,
            batch_size=batch_size,
            max_length=max_length
        )
    else:
        mutation_losses = torch.tensor([])
    
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
    from src.data_utils.RobustSCoT_datasets import load_dataset
    from transformers import AutoTokenizer, AutoModelForCausalLM

    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
    model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B-Instruct", torch_dtype=torch.bfloat16)

    dataset_name = "STAIR-SFT_diverse"
    _, dataset = load_dataset(dataset_name)
    dataset = dataset[:50]

    # questions = [data['question'] for data in dataset]
    # answers = [data['answer'] for data in dataset]

    # # For testing, we can limit the size or use a small batch
    # questions = questions[:100]
    # answers = answers[:100]

    # losses = calculate_losses_with_dataloader(model, tokenizer, questions, answers, device="cuda:0", batch_size=10, max_length=2048)
    # # losses = calculate_losses_multi_gpu(model, tokenizer, questions, answers, batch_size=10, max_length=2048)
    # print(losses)

    results = probe_operator_gradient_direction(
        model,
        tokenizer,
        dataset,
        batch_size=10,
        max_length=2048,
        device="cuda:0",
        use_amp=True
     )
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
    
    
