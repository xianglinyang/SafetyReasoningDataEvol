'''
General:
1. MMLU
2. Commonsensqa
3. bbh
    date_understanding
    strategyqa
    tracking_shuffled_objects
4. arc

Knowledge reasoning:
1. TriviaQA-Wiki

Reading Comprehension:
1. SQuAD

Code:
1. HumanEval

Math:
1. gsm8k
2. AQuA
3. SVAMP
4. MAWPS: multiarith

Symbolic reasoning:
1. Last Letters
2. Coin Flip

Online Test:
1. Open LLM Leaderboard
2. MT benchmark
'''

# "gsm8k" "aqua" "svamp" "multiarith" "commonsenseqa" "date_understanding" "strategyqa" "tracking_shuffled_objects" "last_letters" "coin_flip"
'''Adapted from https://github.com/kojima-takeshi188/zero_shot_cot'''

# TODO: bbh, triviaqa, squad, humaneval

import re
import json
from datasets import load_dataset
from torch.utils.data import Dataset

from src.llm_zoo.api_base_models import OpenAIModel

def data_reader(dataset_name, split):
    questions = list()
    rationales = list()
    answers = list()

    if dataset_name == "gsm8k":
        gsm8k = load_dataset("gsm8k", "main")
        if split == "train":
            data = gsm8k['train']
        elif split == "test":
            data = gsm8k['test']
        else:
            raise TypeError(f"{dataset_name} does not have {split} splits.")

        for item in data:
            question = item['question']
            rationale, answer = item['answer'].split("#### ")
            questions.append(question)
            rationales.append(rationale)
            answers.append(answer)
    elif dataset_name == "aqua":
        aqua = load_dataset("aqua_rat")
        if split == "train":
            data = aqua['train']
        elif split == "test":
            data = aqua['test']
        elif split == "validation":
            data = aqua['validation']
        else:
            raise TypeError(f"{dataset_name} does not have {split} splits.")

        for item in data:
            question = item['question']
            options = item['options']
            rationale = item['rationale']
            answer = item['correct']

            choice = "(" + "(".join(options)
            choice = choice.replace("(", " (").replace(")", ") ")
            choice = "Answer Choices:" + choice

            questions.append(question.strip()+" "+choice)
            rationales.append(rationale)
            answers.append(answer)
    elif dataset_name == "commonsenseqa":
        csqa = load_dataset("tau/commonsense_qa")
        if split == "train":
            data = csqa['train']
        elif split == "test":
            data = csqa['test']
        elif split == "validation":
            data = csqa['validation']
        else:
            raise TypeError(f"{dataset_name} does not have {split} splits.")
        
        for item in data:
            question = item['question']
            options = item['choices']
            answer = item['answerKey']

            choice = "Answer Choices:"
            for label, text in zip(options["label"], options['text']):
                choice += " ("
                choice += label
                choice += ") "
                choice += text

            questions.append(question.strip() + " " + choice)
            answers.append(answer)

    elif dataset_name == "svamp":
        svamp = load_dataset("ChilleD/SVAMP")
        if split == "train":
            data = svamp['train']
        elif split == "test":
            data = svamp['test']
        else:
            raise TypeError(f"{dataset_name} does not have {split} splits.")
        
        for item in data:
            question = item["Body"]+item['Question']
            # remove decimal representation
            answer = str(item['Answer'])[:-2]
            rationale = item["Equation"]
            
            questions.append(question)
            rationales.append(rationale)
            answers.append(answer)

    elif dataset_name == "multiarith":
        multiarith = load_dataset("ChilleD/MultiArith")
        if split == "train":
            data = multiarith['train']
        elif split == "test":
            data = multiarith['test']
        else:
            raise TypeError(f"{dataset_name} does not have {split} splits.")
        
        for item in data:
            question = item['question']
            answer = item['final_ans']
            questions.append(question)
            answers.append(answer)

    elif dataset_name == "date_understanding":
        date_understanding = load_dataset("hails/bigbench", "date_understanding_zero_shot")
        if split == "train":
            data = date_understanding['train']
        elif split == "validation":
            data = date_understanding['validation']
        else:
            raise TypeError(f"{dataset_name} does not have {split} splits.")
        
        score_to_choices = lambda x: chr(ord('A')+x)
        all_options = ['A', 'B', 'C', 'D', 'E', 'F']
        
        for item in data:
            question = item['inputs']
            options = item['multiple_choice_targets']
            answer = item['multiple_choice_scores']

            question = question.replace("Q: ", "")
            question = question.replace("\nA:", "")
            answer = score_to_choices(answer.index(1))

            choice = "Answer Choices:"
            for label, text in zip(all_options, options):
                choice += " ("
                choice += label
                choice += ") "
                choice += text

            questions.append(question.strip() + " " + choice)
            answers.append(answer)
    
    elif dataset_name == "strategyqa":
        strategyqa = load_dataset("hails/bigbench", "strategyqa_zero_shot")
        if split == "train":
            data = strategyqa['train']
        elif split == "validation":
            data = strategyqa['validation']
        else:
            raise TypeError(f"{dataset_name} does not have {split} splits.")
        
        score_to_target = lambda x: "No" if x.index(1) else 'Yes'

        for item in data:
            inputs = item['inputs']
            target = item['targets'][0]
            inputs = inputs.replace("Q: ", "")
            inputs = inputs.replace("\nA:", "")

            question = target+inputs
            answer = score_to_target(item['multiple_choice_scores'])

            questions.append(question)
            answers.append(answer)

    elif dataset_name == "shuffled_objects":
        tso = load_dataset("hails/bigbench", "tracking_shuffled_objects_zero_shot")
        if split == "train":
            data = tso['train']
        elif split == "validation":
            data = tso['validation']
        else:
            raise TypeError(f"{dataset_name} does not have {split} splits.")
        
        score_to_choices = lambda x: chr(ord('A')+x)
        all_options = ['A', 'B', 'C', 'D', 'E']
        
        for item in data:
            question = item['inputs']
            options = item['multiple_choice_targets']
            answer = item['multiple_choice_scores']

            answer = score_to_choices(answer.index(1))

            choice = "Answer Choices:"
            for label, text in zip(all_options, options):
                choice += " ("
                choice += label
                choice += ") "
                choice += text

            questions.append(question.strip() + " " + choice)
            answers.append(answer)
    
    elif dataset_name == "mmlu":
        ds = load_dataset("cais/mmlu", "all")
        # load_dataset("cais/mmlu", "abstract_algebra")
        if split == "test":
            data = ds['test']
        elif split == "validation":
            data = ds['validation']
        elif split == "dev":
            data = ds['dev']
        else:
            raise TypeError(f"{dataset_name} does not have {split} splits.")
        
        score_to_choices = lambda x: chr(ord('A')+x)
        all_options = ['A', 'B', 'C', 'D']
        
        for item in data:
            question = item['question']
            choices = item['choices']
            answer = item['answer']

            answer = score_to_choices(int(answer))

            choice = "Answer Choices:"
            for label, text in zip(all_options, choices):
                choice += " ("
                choice += label
                choice += ") "
                choice += text

            questions.append(question.strip() + " " + choice)
            answers.append(answer)
        
        # # follow dolly format
        # for item in data:
        #     question = item['question']
        #     choices = item['choices']
        #     answer = item['answer']

        #     question = question + " " + ",".join(choices)
        #     questions.append(question)
        #     answers.append(choices[answer])

    
    elif dataset_name in ("coin_flip", "last_letters"):
      dataset_path = f"../dataset/{dataset_name}/{dataset_name}.json"
      with open(dataset_path) as f:
        json_data = json.load(f)
        json_data = json_data["examples"]
        for line in json_data:
          q = line["question"]
          a = line["answer"]
          questions.append(q)
          answers.append(a)
    else:
        raise ValueError("dataset is not properly defined ...")
    
    q_len_list = list()
    for q in questions:
        q_len_list.append(len(q.split(" ")))
    q_len_mean = sum(q_len_list) / len(q_len_list)
    
    print("dataset : {}".format(dataset_name))
    print("data size : {}".format(len(answers)))
    print("average num of words for each sample : {}".format(q_len_mean))
    
    return questions, rationales, answers


class ReasoningDataset(Dataset):
    def __init__(self, dataset_name, split):
        super().__init__()
        self.dataset_name = dataset_name
        self.questions, self.rationales, self.answers = data_reader(self.dataset_name, split)
        self.len = len(self.questions)

        if len(self.rationales) == 0:
            self.rationales = [None]*self.len
        
    def __len__(self):
        return self.len
    
    def __getitem__(self, index):
        question = self.questions[index]
        rationale = self.rationales[index]
        answer = self.answers[index]
        return question, rationale, answer


# answer cleansing with 4o-mini or regular expression matching
def answer_cleansing_with_regex(dataset, llm_answer):
    pred = llm_answer.replace("\n\n", "\n").replace("\n", " ").strip()
    
    if dataset in ("aqua", "commonsenseqa"):
        pred = re.findall(r'A|B|C|D|E', pred)
    elif dataset == "bigbench_date":
        pred = re.findall(r'A|B|C|D|E|F', pred)
    elif dataset in ("object_tracking"):
        pred = re.findall(r'A|B|C', pred)
    elif dataset == "mmlu":
        pred = re.findall(r'A|B|C|D', pred)
    elif dataset in ("gsm8k", "multiarith", "svamp"):
        pred = pred.replace(",", "")
        pred = [s for s in re.findall(r'-?\d+\.?\d*', pred)]
    elif dataset in ("strategyqa", "coin_flip"):
        pred = pred.lower()
        pred = re.sub("\"|\'|\n|\.|\s|\:|\,"," ", pred)
        pred = pred.split(" ")
        pred = [i for i in pred if i in ("yes", "no")]
    elif dataset == "last_letters":
        pred = re.sub("\"|\'|\n|\.|\s","", pred)
        pred = [pred]
    else:
        raise ValueError("dataset is not properly defined ...")

    answer = pred[-1] if len(pred) != 0 else "[INVALID]"
    
    # (For arithmetic tasks)
    if answer != "":
        if answer[-1] == ".":
            answer = answer[:-1]
        elif answer.endswith(".00"):
            answer = answer[:-3]

    return answer

async def batch_answer_cleansing_with_llm(dataset, questions, llm_answers, llm_model_name="gpt-4.1-nano"):
    llm = OpenAIModel(model_name=llm_model_name)
    trigger = zero_shot_answer_trigger(dataset)
    prompt = """\
    You are a helpful assistant.
    Your task is to extract the student's answer according to the provided standard from a question and reasoning paragraph, or return "[INVALID]" if there is no identifiable answer in the paragraph.
    Your SHOULD {trigger}.
    You MUST NOT solve the problem directly.

    # Steps
    
    1. Analyze the question and reasoning paragraph.
    2. Check if there is a valid answer present in the reasoning paragraph.
    3. If a valid answer is present, clean the answer to extract the relevant information according to the provided standard.
    4. If there is no valid answer, output "[INVALID]".

    #### Input
    Question: {question}
    Reasoning Paragraph: {answer}

    #### Output
    """
    prompts = [prompt.format(question=question, answer=llm_answer, trigger=trigger) for question, llm_answer in zip(questions, llm_answers)]
    responses = await llm.batch_invoke(prompts)
    return responses

def answer_cleansing_with_llm(dataset, question, answer):
    # use 4o-mini or 7b model to extract the answer from the reasoning output
    # Currently, we use 4o-mini model to extract the answer from the reasoning output
    llm = OpenAIModel(model_name="gpt-4o-mini")

    prompt = f"""\
    You are a helpful assistant.
    Your task is to extract the student's answer according to the provided standard from a question and reasoning paragraph, or return "[INVALID]" if there is no identifiable answer in the paragraph.
    Your SHOULD {zero_shot_answer_trigger(dataset)}.
    You MUST NOT solve the problem directly.

    # Steps
    
    1. Analyze the question and reasoning paragraph.
    2. Check if there is a valid answer present in the reasoning paragraph.
    3. If a valid answer is present, clean the answer to extract the relevant information according to the provided standard.
    4. If there is no valid answer, output "[INVALID]".

    #### Input
    Question: {question}
    Reasoning Paragraph: {answer}

    #### Output
    """
    response = llm.invoke(prompt)
    return response

def batch_gt_answer_cleansing(dataset, answers):
    clean_answers = [gt_answer_cleansing(dataset, answer) for answer in answers]
    return clean_answers

def gt_answer_cleansing(dataset, answer):
    pred = answer
    if dataset in ("gsm8k"):
        pred = pred.replace(",", "")
    return pred

def zero_shot_answer_trigger(dataset):
    trigger = "The answer is"
    if dataset in ("aqua", "commonsenseqa"):
        trigger = "return only ONE CHARACTER (among A through E) as the answer"
    elif dataset == "mmlu":
        trigger = "return only ONE CHARACTER (among A through D) as the answer"
    elif dataset in("gsm8k", "multiarith", "svamp"):
        trigger = "return arabic numerals of the answer"
    elif dataset in ("strategyqa", "coin_flip"):
        trigger = "return the answer (Yes or No)"
    elif dataset in ("bigbench_date"):
        trigger = "return only ONE CHARACTER (among A through F) as the answer"
    elif dataset in ("object_tracking"):
        trigger = "return only ONE CHARACTER (among A through C) as the answer" 
    return trigger

            
