import operator
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from typing import List, Union, Optional, Dict, Any

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



def probe_gradient_direction(target_model, tokenizer, dataset):
    """
    Estimates the direction of steepest loss ascent.
    """
    # operator
    return dataset

if __name__ == "__main__":
    from src.data_utils.RobustSCoT_datasets import load_dataset
    from transformers import AutoTokenizer, AutoModelForCausalLM

    dataset_name = "STAIR-SFT_diverse"
    dataset, _ = load_dataset(dataset_name)

    questions = [data['question'] for data in dataset]
    answers = [data['answer'] for data in dataset]

    # # For testing, we can limit the size or use a small batch
    # questions = questions[:10]
    # answers = answers[:10]

    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
    model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B-Instruct", torch_dtype=torch.bfloat16)
    losses = calculate_losses_with_dataloader(model, tokenizer, questions, answers, device="cuda:0", batch_size=10, max_length=2048)
    print(losses)
