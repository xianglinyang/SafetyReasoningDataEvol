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

class AnswerStrategy:
	def __init__(self):
		self.name = "AnswerStrategy"
		self.template_path = "data/answer_variants/answers"
		self.templates = {}
		self._strategies = __strategies__
	
	def sample_strategy(self):
		"""Randomly select a strategy"""
		strategy = random.choice(self._strategies)
		return strategy
	
	def strategy2answer_template(self, strategy):
		"""Get answer template for the given strategy"""
		if strategy not in self.templates:
			# sample one template from a json file with predefined templates for each strategy
			file_path = os.path.join(self.template_path, f"{strategy.lower()}.json")
			assert os.path.exists(file_path), f"Template file for strategy {strategy} does not exist. Initialize the template from answer_evol.py first."
			with open(file_path, "r") as f:
				templates = json.load(f)
			self.templates[strategy] = templates
		template = random.choice(self.templates[strategy])
		return template
	
	def enrich_answer_with_strategy(self, question, category, strategy):
		"""Enrich answer using specified strategy"""
		template = self.strategy2answer_template(strategy)
		answer_instance = template.format(
			question=question,
			category=category
		)
		return answer_instance
	
	def enrich_answer(self, question, category):
		"""Enrich answer using randomly selected strategy"""
		strategy = self.sample_strategy()
		template = self.strategy2answer_template(strategy)
		answer_instance = template.format(
			question=question,
			category=category
		)
		return answer_instance

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
