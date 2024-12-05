import torch
def get_length(examples):
    lengths = [len(ids) for ids in examples["input_ids"]]

    completion_lens = []
    for labels in examples["labels"]:
        com_len = (torch.tensor(labels) > -1).sum()
        completion_lens.append(com_len)
    return {"length": lengths, "c_length": completion_lens}


def get_data_statistics(lm_datasets):
    """ Get the data statistics of the dataset. """
    if not isinstance(lm_datasets, dict):
        lm_datasets = {"train": lm_datasets}

    for key in lm_datasets:
        dataset = lm_datasets[key]
        data_size = len(dataset)
        dataset = dataset.map(get_length, batched=True)
        lengths = dataset["length"]
        length = sum(lengths) / len(lengths)
        c_lengths = dataset["c_length"]
        c_length = sum(c_lengths) / len(c_lengths)
        print(
            f"[{key} set] examples: {data_size}; # avg tokens: {length}")
        print(
            f"[{key} set] examples: {data_size}; # avg completion tokens: {c_length}")