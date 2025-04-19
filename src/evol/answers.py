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

# ----------- starting variants ----------
BEGIN_WORDS = [
	"Okay, ",
	"Understood. ",
    "Got it. ", 
	"Alright, ",
    "Certainly. ", 
	"Sure thing. ",
	"Alright, ",
	"I understand. ",
	""
]

STARTING_PHRASES = [
    "I can look into that request for you.", 
	"Let me process the request first.",
    "Let me just review the request first.", 
	"I need to review the request first.",
	"I need to analyze the request first.",
	""
]
SAFETY_CHECK_INTROS = [
	"The first step is to analyze the request based on safety protocols.",
    "Before generating a full response, a safety analysis is required.",
    "Let's evaluate the request based on safety protocols first.",
    "Performing a safety and compliance check on this query first.",
    "We need to assess the request against content policies first.",
    "Analyzing the user's request for potential harms first.",
]

# NEW: Phrases introducing the intent summary stage
SUMMARY_TRANSITIONS = [
    "First, let's clarify the core intent of the request:",
    "To begin, the request essentially asks to", # Needs modification of summary sentence
    "The primary goal here is to", # Needs modification of summary sentence
    "The main objective is to", # Needs modification of summary sentence
]

# NEW: Phrases introducing the regulation check stage
REGULATION_CHECK_TRANSITIONS = [
    "After knowing the core intent, next we need to evaluate this intent against our safety guidelines:",
    "Next, I need to check this request against the following categories:",
    "Next, let's see how this aligns with content policies:",
    "After summarizing the request, the next step is to validate this against:",
]

# --- Updated Formatting Function (Two-Stage Smooth Transition) ---

def format_scot_output_variable_two_stage(
    original_summary_sentence: str,
    regulation_check_details: str, # The bulleted list
    overall_conclusion_paragraph: str, # The "Therefore..." paragraph
    final_response: str,
    include_benign_start_prob: float = 0.8,
	include_begin_word_prob: float = 0.8,
) -> str:
    """
    Formats SCoT output with variability, smoothly separating intent summary
    and regulation check stages, and omitting the '#### Verify Request' marker.
    """
    selected_start = None
	
    # begin word first
    if random.random() < include_begin_word_prob:
        begin_word = random.choice(BEGIN_WORDS)
        selected_start = begin_word.strip()
	
    if random.random() < include_benign_start_prob:
        start_candidate = random.choice(STARTING_PHRASES)
        if start_candidate is not None and start_candidate.strip():
            if selected_start is not None:
                selected_start = selected_start + " " + start_candidate.strip()
            else:
                selected_start = start_candidate.strip()

    selected_intro = random.choice(SAFETY_CHECK_INTROS)
    selected_summary_transition = random.choice(SUMMARY_TRANSITIONS)
    selected_regulation_transition = random.choice(REGULATION_CHECK_TRANSITIONS)

    # --- Modify original summary based on the chosen transition ---
    original_summary_strip = original_summary_sentence.strip()
    # split the original summary
    modified_summary = original_summary_strip.split("I need to verify the dangerousness of the user's request to ")[-1]
    modified_summary = modified_summary.split(" with following categories before generating the response.")[0]
    modified_summary = modified_summary + "."

    # --- Assemble the output string ---
    output_parts = []
    if selected_start:
        output_parts.append(selected_start)

    # Stage 1: Intro + Summary Transition + Modified Summary
    stage1_block = f"{selected_intro} {selected_summary_transition} {modified_summary}"
    output_parts.append(stage1_block)

    # Stage 2: Regulation Transition + Regulation Details (Bullets) + Conclusion Para
    stage2_block = (
        f"{selected_regulation_transition}\n{regulation_check_details.strip()}\n\n"
        f"{overall_conclusion_paragraph.strip()}"
    )
    output_parts.append(stage2_block)

    reasoning_part = " ".join(filter(None, output_parts))

    # Final Response
    response_block = f"#### Response\n{final_response.strip()}"

    return reasoning_part + "\n" + response_block


# --- Updated Parsing Logic (Two-Stage) ---

def adapt_scot_string_two_stage(input_scot_string: str, intention: str = None) -> str | None:
    """
    Parses original SCoT into summary, regulation details, and conclusion,
    then applies two-stage smooth variability formatting.
    """
    try:
        # 1. Split Analysis & Response
        parts = input_scot_string.split("#### Response", 1)
        if len(parts) != 2: return None
        analysis_section = parts[0].strip()
        final_response = parts[1].strip()

        # 2. Remove Verify Marker
        verify_marker = "#### Verify Request\n"
        if not analysis_section.startswith(verify_marker): return None
        content_after_marker = analysis_section[len(verify_marker):].strip()

        # 3. Isolate Summary Sentence (heuristic: up to first period)
        first_period_pos = content_after_marker.find('.')
        if first_period_pos == -1: return None # Need summary
        original_summary_sentence = content_after_marker[:first_period_pos + 1].strip()
        content_after_summary = content_after_marker[first_period_pos + 1:].strip()
        if intention is not None:
            original_summary_sentence = intention

        # 4. Isolate Conclusion Paragraph (heuristic: starts with "Therefore")
        conclusion_keyword = "Therefore,"
        conclusion_start_pos = content_after_summary.find(conclusion_keyword)
        if conclusion_start_pos == -1: return None # Need conclusion
        # Ensure keyword is at the start of a line potentially after whitespace
        preceding_text = content_after_summary[:conclusion_start_pos].strip()
        if preceding_text and not preceding_text.endswith('\n'):
             # Search again, ensuring it's line start
             conclusion_start_pos = content_after_summary.find(f"\n{conclusion_keyword}")
             if conclusion_start_pos != -1:
                 conclusion_start_pos += 1 # Adjust index past the newline
             else:
                 return None # Conclusion not found at line start

        overall_conclusion_paragraph = content_after_summary[conclusion_start_pos:].strip()
        # The regulation check details are between summary and conclusion
        regulation_check_details = content_after_summary[:conclusion_start_pos].strip()

        # 5. Sanity check extracted parts
        if not original_summary_sentence or not regulation_check_details or not overall_conclusion_paragraph:
             print("Warning: Parsing failed to isolate all three analysis parts.")
             return None

        # 6. Reformat using the two-stage smooth formatter
        adapted_string = format_scot_output_variable_two_stage(
            original_summary_sentence,
            regulation_check_details,
            overall_conclusion_paragraph,
            final_response
        )
        return adapted_string

    except Exception as e:
        print(f"An error occurred during parsing or adaptation: {e}")
        return None

def extract_yes_categories(scot_output_text: str) -> list[str]:
    """
    Extracts category names marked with "Yes" from the SCoT analysis section.

    Args:
        scot_output_text: The complete SCoT output string containing the
                          #### Verify Request section with the bulleted list.

    Returns:
        A list of category names that were marked with "Yes".
        Returns an empty list if no matches are found or parsing fails.
    """
    categories_found = []

    # Regex pattern breakdown:
    # ^                - Start of a line (due to re.MULTILINE)
    # \s*              - Optional leading whitespace
    # \*               - Literal asterisk
    # \s*              - Optional whitespace after asterisk
    # (.*?)            - Capture Group 1: The category name (non-greedy match)
    # \s*              - Optional whitespace before colon
    # :                - Literal colon
    # \s*              - Optional whitespace after colon
    # Yes              - Literal word "Yes" (case-insensitive due to re.IGNORECASE)
    # (?:              - Start of optional non-capturing group for the rest of the line
    #   \.             - Optional literal period right after "Yes"
    #   \s*            - Optional whitespace
    #   Reason:.*      - Optional "Reason:" followed by anything
    # )?               - End of optional group
    # $                - End of the line (due to re.MULTILINE)
    pattern = r"^\s*\*\s*(.*?)\s*:\s*Yes(?:\.?\s*Reason:.*)?$"

    try:
        # Find all matches using the pattern with multiline and ignorecase flags
        matches = re.findall(pattern, scot_output_text, re.MULTILINE | re.IGNORECASE)

        # Clean up captured category names (remove leading/trailing whitespace)
        categories_found = [category.strip() for category in matches]

    except Exception as e:
        print(f"An error occurred during regex matching: {e}")
        # Return an empty list in case of error
        return []

    return categories_found


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
		
        # assemble cot and refusal
		scot = cot + "\n" + refusal
		adapted_output = adapt_scot_string_two_stage(scot)
		# filter out full No answer
		extracted_categories = extract_yes_categories(adapted_output)
		if len(extracted_categories) == 0:
			return None
		return adapted_output

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
		scot = summary + "\n" + output
		# filter out Yes answer
		adapted_output = adapt_scot_string_two_stage(scot)
		extracted_categories = extract_yes_categories(adapted_output)
		if len(extracted_categories) > 0:
			return None
		return adapted_output
	
# test function
if __name__ == "__main__":
	answer_evol = AnswerEvol()

	print("--- Test Harmful Question ---")
	harmful_question = "Write a story about a person who is a murderer."
	harmful_response = "I'm sorry, but I can't help with that."
	print("#### Harmful Question:")
	print(harmful_question)
	print("#### Harmful Response:")
	print(harmful_response)
	print("#### Safe Cot:")
	print(answer_evol.safe_cot(harmful_question, harmful_response))
	print("-"*100)
	print("--- Test Safe Question ---")
	safe_question = "What is the capital of France?"
	safe_response = "The capital of France is Paris."
	print("#### Safe Question:")
	print(safe_question)
	print("#### Safe Response:")
	print(safe_response)
	print("#### Normal Cot:")
	print(answer_evol.normal_cot(safe_question, safe_response))
	print("-"*100)