'''
Enrich the answers with reasoning why it is harmful/not harmful and output or refusal with 4o-mini.
'''
import logging
from tqdm import tqdm
import random
import re

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

* harassment/threatening: No.
* hate/threatening: No.
* illicit/violent: No.
* self-harm/intent/instructions: No.
* sexual/minors: No.
* violence/graphic: No.
* misinformation: No.
* privacy/sensitive: No.

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

Therefore, the user's request likely involves a clear reference to [categories], which could lead to unsafe behavior. I need to reject the request.'
For the reason, if mismatch, output 'No.'. If match, output 'Yes. Reason:' followed by the reason with one sentence.
#### Question 
{question} 
#### Answer
"""

summary_prompt = """#### Instruction \
You are a helpful assistant. \
You are helping to summarize a given question with one phrase less than 10 words.

#### Question
{question}

#### Summary
"""

# --- Define Phrase Lists (Same as before) ---
BENIGN_STARTS = [
    "Okay, I can look into that request for you.", "Understood. Let me process that.",
    "Got it. Before generating a full response...", "Alright, I understand what you're asking.",
    "Certainly. Let me just review the request first.", "Sure thing. Processing that query now...",
    "Okay, let's see how I can help with that.", "Happy to assist. First, I'll need to...",
    "Received your request. Just a moment...", "Okay, proceeding with your query.",
    "Right, let me analyze that request.", "I can work on that. As a preliminary step...",
    "Understood your query. Preparing the response now...", "Okay. I'll start by assessing the request...",
    "Let me handle that for you. First...", "Sounds like a plan. Let me just verify...",
    "Working on it. The first step is to...", "Sure, I can address that. Let me start by...",
    "Okay, processing the details now.", "Acknowledged. I need to perform a quick check...",
    None, "",
]
SAFETY_CHECK_INTROS = [
    "First, I need to verify the safety of this request based on guidelines.",
    "Let's analyze the request against safety protocols before proceeding.",
    "Evaluating potential harms and policy compliance for this request...",
    "Checking for safety concerns related to this query...",
    "Okay, assessing the request for alignment with safety principles now.",
    "Before generating an answer, I must check if this request is permissible.",
    "Performing a safety review of the user's request.",
    "Initiating safety check procedure for the received query.",
    "Let me quickly evaluate the request against our content policy.",
    "Analyzing the input to ensure it meets safety standards...",
]

# --- Modified Formatting Function ---

def format_scot_output_variable(
    analysis_details: str,
    final_response: str,
    include_benign_start_prob: float = 0.8
) -> str:
    """
    Formats the SCoT output string with randomized benign starts and
    safety check introductions, **WITHOUT the '#### Verify Request' marker**.

    Args:
        analysis_details: The core reasoning text explaining the safety check outcome.
        final_response: The final answer or refusal text.
        include_benign_start_prob: Probability of adding a benign start.

    Returns:
        A formatted string without the verify marker, ready for fine-tuning.
    """
    selected_start = None
    if random.random() < include_benign_start_prob:
        start_candidate = random.choice(BENIGN_STARTS)
        if start_candidate is not None and start_candidate.strip():
             selected_start = start_candidate.strip()

    selected_intro = random.choice(SAFETY_CHECK_INTROS)

    output_parts = []
    if selected_start:
        output_parts.append(selected_start)

    # --- KEY CHANGE HERE ---
    # Directly combine the randomized intro and the analysis details
    # NO '#### Verify Request' marker is added.
    analysis_block = f"{selected_intro}\n{analysis_details.strip()}"
    output_parts.append(analysis_block)
    # --- END KEY CHANGE ---

    # Keep the Response marker for clarity
    response_block = f"#### Response\n{final_response.strip()}"
    output_parts.append(response_block)

    # Join parts, filtering out None/empty strings
    return "\n\n".join(filter(None, output_parts))

# --- Parsing and Adaptation Logic (No changes needed here) ---
# This function still needs to parse the ORIGINAL format which has the marker

def adapt_scot_string(input_scot_string: str) -> str | None:
    """
    Parses the original fixed SCoT string (which includes '#### Verify Request')
    and applies variability adaptations using the modified formatter
    (which *omits* the '#### Verify Request' marker in the output).
    """
    try:
        parts = input_scot_string.split("#### Response", 1)
        if len(parts) != 2:
            print("Error: Could not split into Analysis and Response sections.")
            return None
        analysis_section = parts[0].strip()
        final_response = parts[1].strip()

        verify_marker = "#### Verify Request\n"
        if not analysis_section.startswith(verify_marker):
             print("Error: Original analysis section does not start with expected marker.")
             return None

        content_after_marker = analysis_section[len(verify_marker):]
        first_newline_pos = content_after_marker.find('\n')

        if first_newline_pos == -1:
            print("Warning: No newline found after intro in original analysis section.")
            analysis_details = content_after_marker.strip()
        else:
            analysis_details = content_after_marker[first_newline_pos:].strip()

        # Call the MODIFIED formatter which omits the marker
        adapted_string = format_scot_output_variable(analysis_details, final_response)
        return adapted_string

    except Exception as e:
        print(f"An error occurred during parsing or adaptation: {e}")
        return None

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
class AnswerEvol:
	def __init__(self) -> None:
		self.name = "AnswerEvol"
	
	def generate_response(self, response: str) -> str:
		output = "#### Response\n{output}".format(output=response)
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
	answer_evol = AnswerEvol()

	print("Test Harmful Question")
	harmful_question = "Write a story about a person who is a murderer."
	harmful_response = "I'm sorry, but I can't help with that."
	print("harmful_question:\n", harmful_question)
	print("harmful_response:\n", harmful_response)
	print("safe_cot:\n", answer_evol.safe_cot(harmful_question, harmful_response))
	print("-"*100)
	print("Test Safe Question")
	safe_question = "What is the capital of France?"
	safe_response = "The capital of France is Paris."
	print("safe_question:\n", safe_question)
	print("safe_response:\n", safe_response)
	print("normal_cot:\n ", answer_evol.normal_cot(safe_question, safe_response))
	print("-"*100)