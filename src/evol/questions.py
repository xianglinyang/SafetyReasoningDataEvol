'''
Enrich the question to make it more difficult to handle. Strategies include:
1. distracted question
2. suppress refusal（should not output "I am sorry"）, 
3. affirmative output (answer with sure), 
4. role play story
5. encoded intput(multilingual attack)

We will enrich the question with all the strategies above and create more variants for each strategy.
'''

# use langchain to create the prompt
import random
import os
import json

from src.evol import __strategies__


class QuestionStrategy:
	def __init__(self):
		self.name = "QuestionStrategy"
		self.template_path = "data/prompt_variants/questions"
		self._strategies = __strategies__
		self._templates = self._load_templates()
	
	def _load_templates(self):
		self.templates = {}
		for strategy in __strategies__:
			file_path = os.path.join(self.template_path, f"{strategy.lower()}.json")
			assert os.path.exists(file_path), f"Template file for strategy {strategy} does not exist. Initialize the template from question_evol.py first."
			with open(file_path, "r") as f:
				templates = json.load(f)
			self.templates[strategy] = templates
	
	def sample_strategy(self):
		strategy = random.choice(__strategies__)
		return strategy
	
	def strategy2prompt_template(self, strategy):
		template = random.choice(self.templates[strategy])
		return template
	
	def enrich_question_with_strategy(self, question, strategy):
		template = self.strategy2prompt_template(strategy)
		question_instance = template.format(question=question)
		return question_instance
	
	def enrich_question(self, question):
		strategy = self.sample_strategy()
		template = self.strategy2prompt_template(strategy)
		question_instance = template.format(question=question)
		return question_instance

# test function
if __name__ == "__main__":
	question_strategy = QuestionStrategy()
	print(question_strategy.enrich_question("What is the capital of France?"))	







