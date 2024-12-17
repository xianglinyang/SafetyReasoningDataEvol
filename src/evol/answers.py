'''
Enrich the answers with reasoning why it is harmful/not harmful and output or refusal.
The format contains (customized by strategy):
1. rephrase content/question
2. ask is it harmful+reasoning why it is/not harmful 
3. output or refusal
'''

import random
import os
import json
from src.evol.questions import __strategies__

# ----------Format----------
refusal_format = """\
#### Verify Request
This is a request asking to output involving {category}, which would lead to unsafe content and is prohibited.

#### Respond
{output}
"""

normal_format = """\
#### Verify Request
This is a request asking to output involving {category}.

#### Respond
{output}
"""


# ----------Answer Strategy----------

class AnswerStrategy:
	def __init__(self) -> None:
		self.name = "AnswerStrategy"
		self.refusal_format = refusal_format
		self.normal_format = normal_format
	
	def enrich_refusal(self, category: str, refusal_output: str) -> str:
		return refusal_format.format(category=category, refusal_output=refusal_output)
	
	def enrich_normal(self, summary: str, answer: str) -> str:
		return normal_format.format(summary=summary, answer=answer)
	
# First version of answer strategy
# evolve the answer with different strategies
# class AnswerStrategy:
# 	def __init__(self):
# 		self.name = "AnswerStrategy"
# 		self.template_path = "data/prompt_variants/answers"
# 		self._strategies = __strategies__
# 		self._templates = self._load_templates()
	
# 	def _load_templates(self):
# 		self.templates = {}
# 		for strategy in __strategies__:
# 			file_path = os.path.join(self.template_path, f"{strategy.lower()}.json")
# 			assert os.path.exists(file_path), f"Template file for strategy {strategy} does not exist. Initialize the template from answer_evol.py first."
# 			with open(file_path, "r") as f:
# 				templates = json.load(f)
# 			self.templates[strategy] = templates
	
# 	def sample_strategy(self):
# 		"""Randomly select a strategy"""
# 		strategy = random.choice(self._strategies)
# 		return strategy
	
# 	def strategy2answer_template(self, strategy):
# 		"""Get answer template for the given strategy"""
# 		template = random.choice(self.templates[strategy])
# 		return template
	
# 	def answer_template_simplified(self):
# 		file_path = os.path.join(self.template_path, "simplified.json")
# 		assert os.path.exists(file_path), f"Template file for simplified answer does not exist. Initialize the template from answer_evol.py first."
# 		with open(file_path, "r") as f:
# 			templates = json.load(f)
# 		template = random.choice(templates)
# 		return template
	
# 	def enrich_answer_with_strategy(self, question, category, strategy):
# 		"""Enrich answer using specified strategy"""
# 		if category is None:
# 			template = self.answer_template_simplified()
# 			answer_instance = template.format(question=question)
# 		else:
# 			template = self.strategy2answer_template(strategy)
# 			answer_instance = template.format(question=question, category=category)
# 		return answer_instance
	
# 	def enrich_answer(self, question, category):
# 		"""Enrich answer using randomly selected strategy"""
# 		strategy = self.sample_strategy()
# 		answer_instance = self.enrich_answer_with_strategy(question, category, strategy)
# 		return answer_instance


# test function
if __name__ == "__main__":
	answer_strategy = AnswerStrategy()
	print("\nTesting random strategy:")
	print(answer_strategy.enrich_answer(
		"What is the capital of France?",
		"geography"
	))
	
	print("\nTesting all strategies:")
	question = "What is the capital of France?"
	category = "geography"
	for strategy in answer_strategy._strategies:
		print(f"\nStrategy: {strategy}")
		print(answer_strategy.enrich_answer_with_strategy(
			question,
			category,
			strategy
		))
