import json
import re
import asyncio
from tqdm import tqdm

from src.llm_zoo import load_model
from src.utils.common_utils import str2json

PROMPT_TEMPLATE = """\
You are an expert Semantic Consistency Judge. Your task is to evaluate a list of "Candidate Questions" against an "Original Question".

For each candidate, determine if it maintains **Functional Equivalence** with the original.
- **Functional Equivalence** means: If a user asks the Candidate Question, would the exact same answer serve as a perfect answer to the Original Question?

**Input:**
Original Question: "{original_q}"

Candidate List:
{candidates_formatted}

**Evaluation Criteria:**
1. **Strict Intent**: The candidate must ask for the exact same information.
2. **Constraints**: The candidate must NOT add, remove, or alter constraints (e.g., time, location, format, quantity).
3. **Neutrality**: Style changes (slang, formal, verbose) are ACCEPTABLE as long as the meaning is preserved.
4. **Drift**: If the topic shifts or the question becomes open-ended (when original was specific), reject it.

**Output Format:**
Return a JSON object where keys are the id of the candidate questions. For each, provide:
- "Intent similarity": A float between 0.0 and 1.0 indicating how similar the intent of the candidate question is to the original question. Higher is more similar.
- "confidence": A float between 0.0 and 1.0 indicating your certainty.

**Example Output:**
```json
{{
  "1": {{"Intent similarity": 0.98, "confidence": 1.0}},
  "2": {{"Intent similarity": 0.3,  "confidence": 0.95}}
}}
```
**Your JSON Output:**"""



class LLMIntentEvaluator:
    def __init__(self, llm_client):
        """
        Args:
            llm_caller_fn: A function `fn(prompt) -> str` that calls your LLM API 
                           (e.g., GPT-4o, Llama-3-70B).
        """
        self.llm_client = llm_client

    def judge_candidates(self, original_q, candidates):
        """
        Args:
            original_q: str
            candidates: List[str]
            confidence_threshold: float (only keep YES if confidence >= this)
            
        Returns:
            valid_candidates: List[str]
            audit_logs: List[dict]
        """
        # 1. Format candidates into a numbered list
        # "1. Candidate A\n2. Candidate B..."
        candidates_formatted = "\n".join([f"{i+1}. {c}" for i, c in enumerate(candidates)])
        
        # 2. Build Prompt
        prompt = PROMPT_TEMPLATE.format(
            original_q=original_q,
            candidates_formatted=candidates_formatted
        )
        
        # 3. Call LLM (Expect JSON)
        # Ensure your LLM caller sets temperature=0.0 for consistency
        response_str = self.llm_client.invoke(prompt)
        results = str2json(response_str)
        return results
    
    async def async_judge_candidates(self, original_qs, candidate_lists):
        """
        Args:
            original_qs: List[str]
            candidate_lists: List[List[str]]
            confidence_threshold: float (only keep YES if confidence >= this)
            
        Returns:
            valid_candidates: List[str]
            audit_logs: List[dict]
        """
        candidates_formatted = []
        for original_q, candidates in zip(original_qs, candidate_lists):
            candidates_formatted.append(f"Original Question: {original_q}\nCandidate List: {candidates}")
        prompts = [
            PROMPT_TEMPLATE.format(
                original_q=original_q,
                candidates_formatted=candidates_formatted
            )
            for original_q, candidates_formatted in zip(original_qs, candidates_formatted)
        ]
        responses = await self.llm_client.batch_invoke(prompts)
        results = [str2json(response) for response in responses]
        return results
    
    def filter_candidates(self, question_qs, candidate_lists, results, similarity_threshold=0.9, confidence_threshold=0.9):
        """
        Args:
            question_qs: List[str]
            candidate_lists: List[List[str]]
            results: List[dict]
            confidence_threshold: float (only keep YES if confidence >= this)
        """
        valid_candidates = []
        for question_q, candidate_list, result in tqdm(zip(question_qs, candidate_lists, results)):
            candidates = []
            for id, result_dict in result.items():
                candidate = candidate_list[int(id)-1]
                if float(result_dict["Intent similarity"]) >= similarity_threshold and float(result_dict["confidence"]) >= confidence_threshold:
                    candidates.append(candidate)
            valid_candidates.append({
                question_q: candidates,
            })
        return valid_candidates



async def main():
    llm_client = load_model("openai/gpt-4.1")
    evaluator = LLMIntentEvaluator(llm_client)

    original = ["What is the capital of France?", "What is the capital of China?"]
    candidates = [
        ["Tell me the capital city of France.", "What is the capital of France in 1400?", "Is Paris the capital of France?"],
        ["Tell me the capital city of China.", "What is the capital of China in 1400?", "Is Beijing the capital of China?"]
    ]

    results = await evaluator.async_judge_candidates(original, candidates)
    print("Results:", results)
    print("--------------------------------")
    valid_ones = evaluator.filter_candidates(original, candidates, results, similarity_threshold=0.9, confidence_threshold=0.9)
    print("Valid Candidates:", valid_ones)

if __name__ == "__main__":
    asyncio.run(main())