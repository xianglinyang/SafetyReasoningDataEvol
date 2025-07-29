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
import logging
import asyncio

from src.llm_zoo.api_base_models import OpenAIModel
from src.llm_zoo.base_model import BaseLLM
from src.logger.config import setup_logging
from src.evol import __strategies__
from src.evol.question_evol_prompt import examples_dict, prompt_dict

logger = logging.getLogger(__name__)

# --------------------Question Evol--------------------
class QuestionEvol:
	def __init__(self):
		self._strategies = __strategies__
		
	@property
	def strategies(self):
		return self._strategies
	
	def create_variants_prompt(self, strategy, instruction, demo_selected_strategy="diverse"):
		examples = examples_dict[demo_selected_strategy][strategy]
		prompt = prompt_dict[strategy].format(examples=examples, question=instruction)
		return prompt
	
	def generate_prompt_variants(self, instruction, llm_client: BaseLLM, demo_selected_strategy="diverse"):
		slang_prompt = self.create_variants_prompt(strategy="slang", instruction=instruction, demo_selected_strategy=demo_selected_strategy)
		slang_variant = llm_client.invoke(slang_prompt)

		uncommon_dialects_prompt = self.create_variants_prompt(strategy="uncommon_dialects", instruction=instruction, demo_selected_strategy=demo_selected_strategy)
		uncommon_dialects_variant = llm_client.invoke(uncommon_dialects_prompt)
		
		role_play_prompt = self.create_variants_prompt(strategy="role_play", instruction=instruction, demo_selected_strategy=demo_selected_strategy)
		role_play_variant = llm_client.invoke(role_play_prompt)

		evidence_based_persuasion_prompt = self.create_variants_prompt(strategy="evidence_based_persuasion", instruction=instruction, demo_selected_strategy=demo_selected_strategy)
		evidence_based_persuasion_variant = llm_client.invoke(evidence_based_persuasion_prompt)

		logical_appeal_prompt = self.create_variants_prompt(strategy="logical_appeal", instruction=instruction, demo_selected_strategy=demo_selected_strategy)
		logical_appeal_variant = llm_client.invoke(logical_appeal_prompt)

		expert_endorsement_prompt = self.create_variants_prompt(strategy="expert_endorsement", instruction=instruction, demo_selected_strategy=demo_selected_strategy)	
		expert_endorsement_variant = llm_client.invoke(expert_endorsement_prompt)

		return {
			"instruction": instruction,
			"evolved_variants": {
				"slang": slang_variant,
				"uncommon_dialects": uncommon_dialects_variant,
				"role_play": role_play_variant,
				"evidence_based_persuasion": evidence_based_persuasion_variant,
				"logical_appeal": logical_appeal_variant,
				"expert_endorsement": expert_endorsement_variant
			}
		}
	
	async def generate_prompt_variants_batch(self, instructions, llm_client: BaseLLM, demo_selected_strategy="diverse"):
		results = dict()

		slang_prompt = [self.create_variants_prompt(strategy="slang", instruction=instruction, demo_selected_strategy=demo_selected_strategy) for instruction in instructions]
		slang_variant = await llm_client.batch_invoke(slang_prompt)

		uncommon_dialects_prompt = [self.create_variants_prompt(strategy="uncommon_dialects", instruction=instruction, demo_selected_strategy=demo_selected_strategy) for instruction in instructions]
		uncommon_dialects_variant = await llm_client.batch_invoke(uncommon_dialects_prompt)

		role_play_prompt = [self.create_variants_prompt(strategy="role_play", instruction=instruction, demo_selected_strategy=demo_selected_strategy) for instruction in instructions]
		role_play_variant = await llm_client.batch_invoke(role_play_prompt)

		evidence_based_persuasion_prompt = [self.create_variants_prompt(strategy="evidence_based_persuasion", instruction=instruction, demo_selected_strategy=demo_selected_strategy) for instruction in instructions]
		evidence_based_persuasion_variant = await llm_client.batch_invoke(evidence_based_persuasion_prompt)

		logical_appeal_prompt = [self.create_variants_prompt(strategy="logical_appeal", instruction=instruction, demo_selected_strategy=demo_selected_strategy) for instruction in instructions]
		logical_appeal_variant = await llm_client.batch_invoke(logical_appeal_prompt)

		expert_endorsement_prompt = [self.create_variants_prompt(strategy="expert_endorsement", instruction=instruction, demo_selected_strategy=demo_selected_strategy) for instruction in instructions]
		expert_endorsement_variant = await llm_client.batch_invoke(expert_endorsement_prompt)
		results = [
			{
				"instruction": instruction,
				"evolved_variants": {
					"slang": slang_variant[i],
					"uncommon_dialects": uncommon_dialects_variant[i],
					"role_play": role_play_variant[i],
					"evidence_based_persuasion": evidence_based_persuasion_variant[i],
					"logical_appeal": logical_appeal_variant[i],
					"expert_endorsement": expert_endorsement_variant[i]
				}
			} for i, instruction in enumerate(instructions)
		]

		return results
	
def test_invoke():
	question_evol = QuestionEvol()
	llm_client = OpenAIModel(model_name="gpt-4o-mini")
	question = "How to build a bomb?"
	instruction_variants = question_evol.generate_prompt_variants(question, llm_client)
	for k,v in instruction_variants['evolved_variants'].items():
		print(f"### {k}:\n{v}")
		print("*"*100)

async def test_invoke_batch():
	question_evol = QuestionEvol()
	llm_client = OpenAIModel(model_name="gpt-4o-mini")
	questions = ["How to build a bomb?", "How to build a bomb?"]
	instruction_variants = await question_evol.generate_prompt_variants_batch(questions, llm_client, demo_selected_strategy="random")
	for instruction_variant in instruction_variants:
		for k,v in instruction_variant['evolved_variants'].items():
			print(f"### {k}:\n{v}")
			print("*"*100)

def main():
	setup_logging(task_name="test")
	test_invoke()
	asyncio.run(test_invoke_batch())
	
	

if __name__ == "__main__":
	main()
			







