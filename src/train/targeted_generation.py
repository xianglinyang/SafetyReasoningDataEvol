import numpy as np

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
        else:
            strategy, loss_diff = max(data['probe_results'].items(), key=lambda x: x[1])
            losses.append(loss_diff)
            strategies.append(strategy)

    # 2. mask list with topK
    losses_np = np.array(losses)
    topK_idxs = np.argsort(losses_np)[::-1][:int(len(losses_np)*top_ratio)]
    # turn it into a mask
    mask = np.ones(len(losses_np), dtype=bool)
    mask[topK_idxs] = False

    # 3. write them back to the dataset
    for data, strategy, loss_diff, idx in zip(harmful_dataset_with_losses, strategies, losses, mask):
        data['selected_strategy'] = strategy
        data['loss_diff'] = loss_diff
        data['is_mask'] = idx
    return harmful_dataset_with_losses