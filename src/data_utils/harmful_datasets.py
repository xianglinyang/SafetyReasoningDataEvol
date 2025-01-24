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
    "sorrybench",
    "advbench",
    "Malicious Instruct",
    "circuitbreaker"
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

def data_reader(dataset_name, split):
    assert dataset_name in __datasets__, f"{dataset_name} is not supported yet"

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
        if split == "train":
            data = dataset['train']
        else:
            raise TypeError(f"{dataset_name} does not have {split} splits.")
        
        for item in data:
            prompt_style = item['prompt_style']
            # current only support base style, other styles are linguistic mutations
            if prompt_style == "base":
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

    return questions, categories


class HarmfulDataset(Dataset):
    def __init__(self, dataset_name, split):
        super().__init__()
        self.dataset_name = dataset_name
        self.questions, self.categories = data_reader(self.dataset_name, split)
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
    