'''
加载 probe 结果并生成用于训练的 dataset
'''
import numpy as np
import random
import json
from typing import List, Dict, Any


def load_probe_results(probe_results_path: str) -> List[Dict[str, Any]]:
    """
    从 JSON 文件加载 probe 结果
    
    Args:
        probe_results_path: probe 结果 JSON 文件路径
        
    Returns:
        List of dicts with keys: question, answer, original_loss, probe_results
    """
    with open(probe_results_path, 'r', encoding='utf-8') as f:
        results = json.load(f)
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


if __name__ == "__main__":
    # 测试用法
    import sys
    if len(sys.argv) < 2:
        print("Usage: python load_probe_results.py <probe_results.json>")
        sys.exit(1)
    
    probe_results = load_probe_results(sys.argv[1])
    print(f"Loaded {len(probe_results)} probe results")
    
    dataset = select_strategies(probe_results, top_ratio=0.1)
    print(f"Generated dataset with {len(dataset)} samples")
    print(f"Samples with is_mask=True: {sum(1 for d in dataset if d['is_mask'])}")
    
    # 打印前几个样本
    print("\nFirst 3 samples:")
    for i, data in enumerate(dataset[:3]):
        print(f"\nSample {i+1}:")
        print(f"  Question: {data['question'][:80]}...")
        print(f"  Selected strategy: {data['selected_strategy']}")
        print(f"  Is mask: {data['is_mask']}")