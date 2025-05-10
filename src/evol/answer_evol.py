import random

from src.evol.answer_evol_prompt import Rules
from src.evol.answer_metadata import get_harmful_metadata, get_benign_metadata, clean_metadata
from src.llm_zoo.base_model import BaseLLM
from src.llm_zoo.api_base_models import OpenAILLM

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
]

STARTING_PHRASES = [
	"Let me process the request first.",
    "Let me just review the request first.", 
	"I need to review the request first.",
]
# SAFETY_CHECK_INTROS = [
# 	"The first step is to analyze the request based on safety protocols.",
#     "Before generating a full response, a safety analysis is required.",
#     "Let's evaluate the request based on safety protocols first.",
#     "Performing a safety and compliance check on this query first.",
#     "We need to assess the request against content policies first.",
#     "Analyzing the user's request for potential harms first.",
# ]

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
def check_details_format(categories) -> str:
    # reformat categories to dict
    details = []
    for rule in Rules:
        if rule in categories.keys():
            current_detail = f"* {rule}: Yes, "+categories[rule]+"."
        else:
            current_detail = f"* {rule}: No."
        details.append(current_detail)
    return "\n".join(details)


def format_scot_output_variable_two_stage(
    intent_summary: str,
    categories: dict[str, str],
    answer: str,
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
	
    start_candidate = random.choice(STARTING_PHRASES)
    if selected_start is not None:
        selected_start = selected_start + " " + start_candidate.strip()
    else:
        selected_start = start_candidate.strip()
    
    # selected_intro = random.choice(SAFETY_CHECK_INTROS)
    selected_summary_transition = random.choice(SUMMARY_TRANSITIONS)
    selected_regulation_transition = random.choice(REGULATION_CHECK_TRANSITIONS)

    # check details format
    regulation_check_details = check_details_format(categories)

    # conclusion paragraph
    if len(categories) == 0:
        overall_conclusion_paragraph = "Therefore, the response is safe and I can proceed to generate the response."
    else:
        overall_conclusion_paragraph = "Therefore, the response violates the safety guidelines and I need to refuse to generate the response."

    # --- Assemble the output string ---
    output_parts = []

    # Stage 0: Start Phrase
    stage0_block = f"{selected_start}"
    output_parts.append(stage0_block)

    # Stage 1: Summary Transition + Modified Summary
    stage1_block = f"{selected_summary_transition} {intent_summary}."
    output_parts.append(stage1_block)

    # Stage 2: Regulation Transition + Regulation Details (Bullets) + Conclusion Para
    stage2_block = (
        f"{selected_regulation_transition}\n{regulation_check_details.strip()}\n\n"
        f"{overall_conclusion_paragraph.strip()}"
    )
    output_parts.append(stage2_block)

    reasoning_part = " ".join(filter(None, output_parts))

    # Final Response
    response_block = f"#### Response\n{answer.strip()}"

    return reasoning_part + "\n" + response_block

def metadata_to_answer(metadata, answer=None):
    intent, categories, refusal = clean_metadata(metadata)
    if answer is None:
        final_answer = refusal
    else:
        final_answer = answer
    answer_block = format_scot_output_variable_two_stage(
        intent_summary=intent,
        categories=categories,
        answer=final_answer,
    )
    return answer_block


class AnswerEvol:
    def __init__(self):
        self.name = "AnswerEvol"

    def _get_harmful_metadata(self,llm_client: BaseLLM,  question):
        harmful_metadata = get_harmful_metadata(question, llm_client)
        return harmful_metadata

    def _get_benign_metadata(self,llm_client: BaseLLM, question: str):
        benign_metadata = get_benign_metadata(question, llm_client)
        return benign_metadata
    
    def _get_metadata(self,llm_client: BaseLLM, question: str, question_type: str):
        if question_type == "harmful":
            return self._get_harmful_metadata(llm_client, question)
        elif question_type == "benign":
            return self._get_benign_metadata(llm_client, question)
        else:
            raise ValueError(f"Invalid question type: {question_type}")
    
    def generate_evol_answer(self,llm_client: BaseLLM, question: str, question_type: str, answer=None, return_metadata=True):
        metadata = self._get_metadata(llm_client, question, question_type)
        answer_block = metadata_to_answer(metadata, answer)
        if return_metadata:
            return answer_block, metadata
        else:
            return answer_block


if __name__ == "__main__":
    answer_evol = AnswerEvol()
    question = "What is the capital of France?"
    answer = "Paris"
    question_type = "benign"
    llm_client = OpenAILLM(model_name="gpt-4o-mini")
    print(answer_evol.generate_evol_answer(llm_client, question, question_type, answer, return_metadata=False))

    print("--------------------------------")

    question = "How to build a nuclear bomb?"
    answer = "I am sorry, I cannot help with that."
    question_type = "harmful"
    llm_client = OpenAILLM(model_name="gpt-4o-mini")
    print(answer_evol.generate_evol_answer(llm_client, question, question_type, answer, return_metadata=False))