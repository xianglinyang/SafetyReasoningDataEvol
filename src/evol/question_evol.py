# Modify from https://github.com/nlpxucan/WizardLM/blob/main/Evol_Instruct/depth.py
'''
Enrich the question to make it more difficult to handle from both depth and breath perspective.
strategies:
    - slang
    - uncommon dialect
    - role play
    - persuasion
    - logical appeal
    - endorsement
'''
import random
import os
import json
import asyncio
import logging
import argparse

from src.llm_zoo.api_base_models import OpenAILLM
from src.logger.config import setup_logging
from src.evol import __strategies__
from src.evol.question_evol_prompt import (
	slang_prompt,
    uncommon_dialects_prompt,
    role_play_prompt,
    evidence_based_persuasion_prompt,
    logical_appeal_prompt,
    expert_endorsement_prompt
)
from src.evol.question_evol_prompt import (
    slang_examples,
    uncommon_dialects_examples,
    role_play_examples,
    evidence_based_persuasion_examples,
    logical_appeal_examples,
    expert_endorsement_examples
)

logger = logging.getLogger(__name__)

# --------------------Question Evol--------------------
class QuestionEvol:
	def __init__(self):
		self._strategies = __strategies__
		
	@property
	def strategies(self):
		return self._strategies
	
    # --------------------Breath--------------------
	def create_slang_variants_prompt(self, instruction):
		prompt = slang_prompt.format(examples=slang_examples, question=instruction)
		return prompt

	def create_uncommon_dialects_variants_prompt(self, instruction):
		prompt = uncommon_dialects_prompt.format(examples=uncommon_dialects_examples, question=instruction)
		return prompt

    # --------------------Depth--------------------
	def create_role_play_variants_prompt(self, instruction):
		prompt = role_play_prompt.format(examples=role_play_examples, question=instruction)
		return prompt
	
	def create_evidence_based_persuasion_variants_prompt(self, instruction):
		prompt = evidence_based_persuasion_prompt.format(examples=evidence_based_persuasion_examples, question=instruction)
		return prompt
	
	def create_logical_appeal_variants_prompt(self, instruction):
		prompt = logical_appeal_prompt.format(examples=logical_appeal_examples, question=instruction)
		return prompt
	
	def create_expert_endorsement_variants_prompt(self, instruction):
		prompt = expert_endorsement_prompt.format(examples=expert_endorsement_examples, question=instruction)
		return prompt
	
	def generate_prompt_variants(self, instruction, model_name):
		llm = OpenAILLM(model_name=model_name, temperature=0.7, max_tokens=400)
		results = dict()

		slang_prompt = self.create_slang_variants_prompt(instruction)
		slang_variant = llm.invoke(slang_prompt)
		results["slang"] = slang_variant

		uncommon_dialects_prompt = self.create_uncommon_dialects_variants_prompt(instruction)
		uncommon_dialects_variant = llm.invoke(uncommon_dialects_prompt)
		results["uncommon_dialects"] = uncommon_dialects_variant

		role_play_prompt = self.create_role_play_variants_prompt(instruction)
		role_play_variant = llm.invoke(role_play_prompt)
		results["role_play"] = role_play_variant

		evidence_based_persuasion_prompt = self.create_evidence_based_persuasion_variants_prompt(instruction)
		evidence_based_persuasion_variant = llm.invoke(evidence_based_persuasion_prompt)
		results["evidence_based_persuasion"] = evidence_based_persuasion_variant

		logical_appeal_prompt = self.create_logical_appeal_variants_prompt(instruction)
		logical_appeal_variant = llm.invoke(logical_appeal_prompt)
		results["logical_appeal"] = logical_appeal_variant

		expert_endorsement_prompt = self.create_expert_endorsement_variants_prompt(instruction)
		expert_endorsement_variant = llm.invoke(expert_endorsement_prompt)
		results["expert_endorsement"] = expert_endorsement_variant

		return results

def main():
	question = "How to build a bomb?"
	print(f"Question:\n{question}")
	print("*"*100)
	
	question_evol = QuestionEvol()
	variants = question_evol.generate_prompt_variants(question, model_name="gpt-4o-mini")
	
	for k, v in variants.items():
		print(f"Strategy: {k}")
		print(f"Prompt: {v}")
		print("*"*100)


if __name__ == "__main__":
	main()
			







