'''
This file is used to load harmful datasets and evaluate them.
Harmful datasets:
1. jailbreakbench
2. harmbench
3. sorrybench
4. advbench
5. Malicious Instruct
'''
import json
from datasets import load_dataset
from torch.utils.data import Dataset
__datasets__ = [
    "jailbreakbench",
    "harmbench",
    "harmbench_attack",
    "sorrybench",
    "advbench",
    "Malicious Instruct",
    "circuitbreaker"
    "renellm"
    "xstest"
]

sorrybench_categories = [
    "Personal Insult Words",
    "Social-group Insult Words",
    "Threatening Words",
    "Lewd and Obscene Words",
    "Libelous Words",
    "Self-Harm",
    "Violent Crimes",
    "Harassment",
    "Sexual Crimes",
    "Property Crimes",
    "Public Order Crimes",
    "Impersonation",
    "System Intrusion",
    "Malware",
    "Fraud",
    "Financial Crimes",
    "IP Infringement",
    "PII Violations",
    "Illegal Crafting",
    "Terrorism",
    "Child-related Crimes",
    "Animal-related Crimes",
    "Environmental Crimes",
    "Evasion of Law",
    "Human Elimination",
    "Advice on Adult Content",
    "Sexual Explicit Content",
    "Non-sexual Explicit Content",
    "Fake News",
    "False Advertising",
    "Discrimination",
    "Military Use",
    "Political Belief",
    "Ethical Belief",
    "Religion",
    "Social Stereotypes",
    "Extremist Content",
    "Conspiracy Theories",
    "False Common Knowledge",
    "Unhealthy Behaviors",
    "Medical Advice",
    "Financial Advice",
    "Legal Consulting Advice",
    "Governance Decision Advice",
    "Machinery Operation Advice"
]
remove_categories = [
    "1", "4", "5", "10", "11", "15", "16", "17", "22", "23", "28", "31", "32", "33", "34", "35", "41", "42", "43", "44", "45"
]

def data_reader(dataset_name, split, save_dir=None):
    # assert dataset_name in __datasets__, f"{dataset_name} is not supported yet"

    questions = list()
    categories = list()

    if dataset_name == "jailbreakbench":
        dataset = load_dataset("walledai/JailbreakBench")
        if split == "train":
            data = dataset['train']
        else:
            raise TypeError(f"{dataset_name} does not have {split} splits.")
        
        for item in data:
            if item['subset'] == 'harmful':
                prompt = item['prompt']
                behavior = item['behavior']
                questions.append(prompt)
                categories.append(behavior)

    elif dataset_name == "harmbench":
        # standard, copyright, contextual
        dataset = load_dataset("walledai/HarmBench", "standard")
        if split == "train":
            data = dataset['train']
        else:
            raise TypeError(f"{dataset_name} does not have {split} splits.")
        
        for item in data:
            prompt = item['prompt']
            category = item['category']
            questions.append(prompt)
            categories.append(category)

    elif dataset_name == "sorrybench":
        dataset = load_dataset("sorry-bench/sorry-bench-202406")
        # remove those categories not in our categories
        data = [item for item in dataset['train'] if item['prompt_style'] == split and item['category'] not in remove_categories]
        for item in data:
            prompt = item['turns'][0]
            category = sorrybench_categories[int(item['category'])-1]
            questions.append(prompt)
            categories.append(category)

    elif dataset_name == "advbench":
        dataset = load_dataset("walledai/AdvBench")
        if split == "train":
            data = dataset['train']
        else:
            raise TypeError(f"{dataset_name} does not have {split} splits.")
        for item in data:
            prompt = item['prompt']
            questions.append(prompt)
            
    elif dataset_name == "Malicious Instruct":
        dataset = load_dataset("walledai/MaliciousInstruct")
        if split == "train":
            data = dataset['train']
        else:
            raise TypeError(f"{dataset_name} does not have {split} splits.")
        
        for item in data:
            prompt = item['prompt']
            questions.append(prompt)
    
    elif dataset_name == "circuitbreaker":
        assert split in ["train", "val"], f"{dataset_name} does not have {split} splits."
        data_path = f"data/raw/{dataset_name}_{split}.json"
        with open(data_path, 'r') as f:
            data = json.load(f)
        
        for item in data:
            prompt = item["prompt"]
            questions.append(prompt)
    elif dataset_name == "renellm":
        assert split in ["train"], f"{dataset_name} does not have {split} splits."
        dataset = load_dataset("Deep1994/ReNeLLM-Jailbreak")
        data = dataset['train']
        for item in data:
            prompt = item['nested_prompt']
            questions.append(prompt)
    elif dataset_name == "harmbench_attack":
        assert save_dir is not None, "save_dir is required for harmfulbench"
    
        with open(save_dir, 'r') as f:
            data = json.load(f)
        for key, value in data.items():
            questions.append(value[0])
    elif dataset_name == "xstest":
        assert split in ["safe", "unsafe"], f"{dataset_name} does not have {split} splits."
        ds = load_dataset("walledai/XSTest")
        data = ds['test']
        
        for item in data:
            label = item['label']
            if label == split:
                prompt = item['prompt']
                category = item['focus']
                questions.append(prompt)
                categories.append(category)
    else:
        raise ValueError(f"{dataset_name} is not supported yet")

    return questions, categories


class HarmfulDataset(Dataset):
    def __init__(self, dataset_name, split, save_dir=None):
        super().__init__()
        self.dataset_name = dataset_name
        self.questions, self.categories = data_reader(self.dataset_name, split, save_dir)
        self.len = len(self.questions)

    def __len__(self):
        return self.len
    
    def __getitem__(self, index):
        question = self.questions[index]
        if len(self.categories) > 0:
            category = self.categories[index]
        else:
            category = None
        return question, category
    