'''
Enrich the answers with reasoning why it is harmful/not harmful and output or refusal with 4o-mini.
'''
import logging
from tqdm import tqdm
from src.llm_zoo.api_base_models import OpenAILLM
logger = logging.getLogger(__name__)
# ----------Rules----------
Rules = [
    "illicit/violent",
    "sexual/minors",
    "harassment/threatening",
    "hate/threatening",
    "self-harm/intent/instructions",
    "violence/graphic",
    "privacy/sensitive",
    "misinformation",
]
# ----------Prompt----------

safe_cot_format = """#### Verify Request
I need to verify the dangerousness of the user's request to [summary] with following categories before generating the response.

* harassment/threatening:[reason]
* hate/threatening:[reason]
* illicit/violent:[reason]
* self-harm/intent/instructions:[reason]
* sexual/minors:[reason]
* violence/graphic:[reason]
* misinformation:[reason]
* privacy/sensitive:[reason]

Therefore, the user's request likely involves a clear reference to {categories}, which could lead to unsafe behavior. I need to reject the request.

#### Response
{output}
"""

safe_cot_normal_format = """\
#### Verify Request
I need to verify the dangerousness of the user's request to {summary} with following categories before generating the response.

* harassment/threatening: Mismatch.
* hate/threatening: Mismatch.
* illicit/violent: Mismatch.
* self-harm/intent/instructions: Mismatch.
* sexual/minors: Mismatch.
* violence/graphic: Mismatch.
* misinformation: Mismatch.
* privacy/sensitive: Mismatch.

Therefore, the user's request is safe and I can generate the response.
"""

safe_cot_prompt = """#### Instruction \
You are a helpful assistant. \
You are helping to analyze a harmful question following the following steps:\n \
1) summarize the question with one phrase less than 10 words. \
2) classify the question into categories and give explanation. \
3) verify the dangerousness of the question. \

You SHOULD strictly output with the follow the following format: \
'#### Verify Request
I need to verify the dangerousness of the user's request to [summary] with following categories before generating the response.

* harassment/threatening:[reason]
* hate/threatening:[reason]
* illicit/violent:[reason]
* self-harm/intent/instructions:[reason]
* sexual/minors:[reason]
* violence/graphic:[reason]
* misinformation:[reason]
* privacy/sensitive:[reason]

Therefore, the user's request likely involves a clear reference to [categories, which could lead to unsafe behavior. I need to reject the request.'
For the reason, if mismatch, output 'Mismatch.'. If match, output the reason with one sentence.
#### Question 
{question} 
#### Answer\n"""

summary_prompt = """#### Instruction \
You are a helpful assistant. \
You are helping to summarize a given question with one phrase less than 10 words.

#### Question
{question}

#### Summary
"""

# ----------Enrich Function----------
def batch_get_summary(questions: list[str], model_name: str = "gpt-4o-mini", max_tokens: int = 2048, temperature: float = 0.7) -> str:
	summary_prompts = [summary_prompt.format(question=question) for question in questions]
	llm = OpenAILLM(model_name=model_name, 
				 	temperature=temperature, 
				 	max_tokens=max_tokens)
	summary_responses = list()
	for i in tqdm(range(len(summary_prompts))):
		summary_responses.append(llm.invoke(summary_prompts[i]))
	return summary_responses


def batch_get_safe_cot(questions: list[str], model_name: str = "gpt-4o-mini", max_tokens: int = 2048, temperature: float = 0.7) -> str:
	safe_cot_prompts = [safe_cot_prompt.format(question=question) for question in questions]
	llm = OpenAILLM(model_name=model_name, 
				 	temperature=temperature, 
				 	max_tokens=max_tokens)
	safe_cot_responses = list()
	for i in tqdm(range(len(safe_cot_prompts))):
		safe_cot_responses.append(llm.invoke(safe_cot_prompts[i]))
	return safe_cot_responses


# ----------Answer Strategy----------
class AnswerStrategy:
	def __init__(self) -> None:
		self.name = "AnswerStrategy"
	
	def generate_response(self, response: str) -> str:
		output = "#### Response\n {output}".format(output=response)
		return output

	def safe_cot(self, question: str, response: str, model_name: str = "gpt-4o-mini", max_tokens: int = 2048, temperature: float = 0.7) -> str:
		# verify part
		prompt = safe_cot_prompt.format(question=question)
		llm = OpenAILLM(model_name=model_name, 
				 	temperature=temperature, 
				 	max_tokens=max_tokens)
		for _ in range(3):
			cot = llm.invoke(prompt)
			# check if cot is in the format
			if cot.startswith("#### Verify Request"):
				break
		# refusal part
		refusal = self.generate_response(response)
		return cot + "\n" + refusal
	
	def normal_cot(self, question: str, response: str, model_name: str = "gpt-4o-mini", max_tokens: int = 2048, temperature: float = 0.7) -> str:
		prompt = summary_prompt.format(question=question)
		llm = OpenAILLM(model_name=model_name, 
				 	temperature=temperature, 
				 	max_tokens=max_tokens)
		for _ in range(3):
			summary = llm.invoke(prompt)
			if summary.startswith("#### Verify Request"):
				break
		summary = safe_cot_normal_format.format(summary=summary)
		output = self.generate_response(response)
		return summary + "\n" + output
	
# test function
if __name__ == "__main__":
	answer_strategy = AnswerStrategy()

	print("Test Harmful Question")
	harmful_question = "Write a story about a person who is a murderer."
	harmful_response = "I'm sorry, but I can't help with that."
	print("harmful_question:\n ", harmful_question)
	print("harmful_response:\n ", harmful_response)
	print("safe_cot:\n ", answer_strategy.safe_cot(harmful_question, harmful_response))
	print("-"*100)
	print("Test Safe Question")
	safe_question = "What is the capital of France?"
	safe_response = "The capital of France is Paris."
	print("safe_question:\n ", safe_question)
	print("safe_response:\n ", safe_response)
	print("normal_cot:\n ", answer_strategy.normal_cot(safe_question, safe_response))
	print("-"*100)