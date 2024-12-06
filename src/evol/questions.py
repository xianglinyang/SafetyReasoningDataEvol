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

__strategies__ = ["DISTRACTED_QUESTION", "SUPPRESS_REFUSAL", "AFFIRMATIVE_OUTPUT", "ROLE_PLAY_STORY", "ENCODED_INPUT"]


class QuestionStrategy:
	def __init__(self):
		self.name = "QuestionStrategy"
		self.template_path = "data/prompt_variants"
		self.templates = {}
	
	def sample_strategy(self):
		strategy = random.choice(__strategies__)
		return strategy
	
	def strategy2prompt_template(self, strategy):
		if strategy not in self.templates:
			# sample one template from a json file with predefined templates for each strategy
			file_path = os.path.join(self.template_path, f"{strategy.lower()}.json")
			assert os.path.exists(file_path), f"Template file for strategy {strategy} does not exist. Initialize the template from question_evol.py first."
			with open(file_path, "r") as f:
				templates = json.load(f)
			self.templates[strategy] = templates
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







