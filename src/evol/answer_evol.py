"""
Enrich the answers with reasoning why it is harmful/not harmful and output or refusal.
The format contains (customized by strategy):
1. rephrase content/question
2. ask is it harmful+reasoning why it is/not harmful 
3. output or refusal
"""
import random
import os
import json
import asyncio

from src.llm_zoo.api_base_models import OpenAILLM
from src.evol import __strategies__
# --------------------Answer Format Templates--------------------
# TODO: currently we only attach constraints to the question. We can consider using more advanced model to automatically extract constraints from the question.
# 1. Analysis Request
# 2. Harmful Check with reasoning
# 3. Refusal or Output

REPHRASE_FORMAT = """First, let me analyze the request: You are asking me answer ###question### ###constraints###"""

CONSTRAINTS = {
    "DISTRACTED_QUESTION": "while multitaksing on daily tasks.",
    "SUPPRESS_REFUSAL": "and avoid rejected words.",
    "AFFIRMATIVE_OUTPUT": "and start with affirmative tone.",
    "ROLE_PLAY_STORY": "and role-play as a character.",
    "ENCODED_INPUT": "in encoded format.",
}

HARMFUL_CHECK_FORMAT_TEMPLATE = """Next, let's evaluate if this request involves violent or criminal actions before giving the answer. """

REFUSAL_FORMAT = """Therefore, I'm sorry, but I cannot assist with that request."""

# --------------------Prompt Evol--------------------
# Rephrase the prompt with more variations with gpt capability
# 1. variable format remain
# 2. rephrase should be reasonable and must be understood and responded by humans.
# 3. format requirements:
REPHRASE_BASE_INSTRUCTION = """I want you to act as a Prompt Rephraser.\r\n \
	Your goal is to rephrase the #Given Prompt# into a new prompt that has the same meaning as the #Given Prompt# but with a different wording.\r\n \
    Remember not to change the meaning of the original prompt and lose the original context.\r\n \
	The rephrased prompt must be reasonable and must be understood and responded by humans.\r\n \
	Your rewriting should keep the variable format '###variable###' in #The Given Prompt#.\r\n \
	You should try your best not to make the #Rewritten Prompt# become verbose, #Rewritten Prompt# can only add 10 to 20 words into #The Given Prompt#. \r\n \
	'#The Given Prompt#', '#Rewritten Prompt#', 'given prompt' and 'rewritten prompt' are not allowed to appear in #Rewritten Prompt#\r\n" \r\n """


# --------------------Answer Evol--------------------
class AnswerEvol:
    def __init__(self):
        self._strategies = __strategies__
        self.rephrase_format = REPHRASE_FORMAT
        self.harmful_check_format = HARMFUL_CHECK_FORMAT_TEMPLATE
        self.refusal_format = REFUSAL_FORMAT
        self.rephrase_instruction = REPHRASE_BASE_INSTRUCTION
        self.constraints = CONSTRAINTS
    
    @property
    def strategies(self):
        return self._strategies
    
    # --------------------Breath--------------------
    def createRephrasePrompt(self, instruction: str) -> str:
        prompt = self.rephrase_instruction
        prompt += "#Given Prompt#: \r\n {} \r\n".format(instruction)
        prompt += "#Rewritten Prompt#:\r\n"
        return prompt
    
    # --------------------Depth--------------------
    def createReasoningPrompt(self, strategy_name):
        assert strategy_name in self._strategies

        rephrase_prompt = self.rephrase_format.replace("###constraints###", self.constraints[strategy_name])
        reason_prompt = self.harmful_check_format
        reason_prompt += "Since the question is about ###category###, it is prohibited to answer."
        refusal_prompt = self.refusal_format
        prompt = rephrase_prompt + reason_prompt + refusal_prompt
        return prompt
    
    def createAnswerPrompt_wo_category(self, strategy_name):
        assert strategy_name in self._strategies

        rephrase_prompt = self.rephrase_format.replace("###constraints###", self.constraints[strategy_name])
        reason_prompt = self.harmful_check_format
        reason_prompt += "Since the question is harmful and toxic, it is prohibited to answer the question."
        refusal_prompt = self.refusal_format
        prompt = rephrase_prompt + reason_prompt + refusal_prompt
        return prompt
     
    # --------------------Utils--------------------
    def clean_prompt(self, prompt: str, category: bool = True) -> str:
        """Clean and format the prompt"""
        if "###question###" in prompt:
            prompt = prompt.replace("###question###", "{question}")
            if category and "###category###" in prompt:
                prompt = prompt.replace("###category###", "{category}")
                return prompt
            else:
                return prompt
        else:
            return None
    
    async def generate_reasoning_variants_with_strategy(self, strategy_name, model_name, num_variants):
        reasoning_prompt = self.createReasoningPrompt(strategy_name)
        print(reasoning_prompt)
        print("-"*50)

        new_variants = [reasoning_prompt]
        while len(new_variants) < num_variants:
            seed_prompt = random.choice(new_variants)
            rephrase_prompt = self.createRephrasePrompt(seed_prompt)
            llm = OpenAILLM(model_name=model_name, temperature=0.7, max_tokens=200)
            new_prompt = await llm.invoke(rephrase_prompt)
            if self.clean_prompt(new_prompt):
                new_variants.append(new_prompt)
        new_variants = [self.clean_prompt(prompt) for prompt in new_variants]
        return new_variants
    
    async def generate_simplified_answer_variants(self, model_name, num_variants):
        new_variants = []
        for strategy_name in self._strategies:
            reasoning_prompt = self.createAnswerPrompt_wo_category(strategy_name)
            new_variants.append(reasoning_prompt)
            print(reasoning_prompt)
            print("-"*50)
        while len(new_variants) < num_variants:
            seed_prompt = random.choice(new_variants)
            rephrase_prompt = self.createRephrasePrompt(seed_prompt)
            llm = OpenAILLM(model_name=model_name, temperature=0.7, max_tokens=200)
            new_prompt = await llm.invoke(rephrase_prompt)
            if self.clean_prompt(new_prompt, category=False):
                new_variants.append(new_prompt)
                print(len(new_variants))
                print(new_prompt)
        new_variants = [self.clean_prompt(prompt, category=False) for prompt in new_variants]
        return new_variants
        
    def save_answer_variants(self, strategy_name: str, answer_variants: list, file_path="data/prompt_variants/answers"):
        os.makedirs(file_path, exist_ok=True)
        save_file = os.path.join(file_path, f"{strategy_name.lower()}.json")
        if os.path.exists(save_file):
            with open(save_file, "r") as f:
                existing_variants = json.load(f)
            existing_variants.extend(answer_variants)
            answer_variants = existing_variants
        with open(save_file, "w") as f:
            json.dump(answer_variants, f)

if __name__ == "__main__":
    import asyncio
    
    async def main():
        
        num_variants_per_class = 100
        model_name = "gpt-4o"

        answer_evol = AnswerEvol()

        for strategy in ["DISTRACTED_QUESTION", "SUPPRESS_REFUSAL", "AFFIRMATIVE_OUTPUT", "ROLE_PLAY_STORY"]:
            print(f"Generating {num_variants_per_class} variants for {strategy}...")
            answer_variants = await answer_evol.generate_reasoning_variants_with_strategy(strategy, model_name, num_variants_per_class)
            answer_evol.save_answer_variants(strategy, answer_variants)
            print(f"Saved {len(answer_variants)} variants for {strategy}.")
            print("*"*50)

        answer_variants = await answer_evol.generate_simplified_answer_variants(model_name, num_variants_per_class)
        answer_evol.save_answer_variants("simplified", answer_variants)
        print(f"Saved {len(answer_variants)} variants for simplified version.")
        print("*"*50)

    asyncio.run(main())
    