"""
Enrich the answers with reasoning why it is harmful/not harmful and output or refusal.
The format contains:
1. rephrase content/question
2. ask is it harmful+reasoning why it is/not harmful
3. output or refusal
"""

answer_format = f"""Base on the content and question, you are asking me to answer {question}.
Before answering, I need to check if it is harmful or not.
Since the centent of the question belongs to {category}, it is harmful.
Therefore, I am sorry that I cannot answer that.
"""

# Format 1
answer_format_1 = f"""You have asked about {question}.
To ensure safety, I must determine if the content is harmful.
Given that the question pertains to {category}, it is deemed harmful.
Regrettably, I am unable to provide an answer."""

# Format 2
answer_format_2 = f"""The question posed is: {question}.
I need to evaluate the potential harm of the content.
As the question is categorized under {category}, it is considered harmful.
Thus, I must refrain from answering."""

# Format 3
answer_format_3 = f"""Your inquiry is: {question}.
Before proceeding, I must assess the harm potential.
Since the question falls under the {category} category, it is harmful.
Consequently, I cannot provide an answer."""

def enrich_answers(question, category, content):
    prompt = f"based on the answer format {content}, suggest more format like this"
    return prompt
