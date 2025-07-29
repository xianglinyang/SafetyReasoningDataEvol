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
    "copyrighted",
]
# ----------System Prompt----------
HARMFUL_SYSTEM_PROMPT = """You are a expert in analyzing harmful questions. Your task is to analyze the harmful question and output the detailed analysis of the harmful question and response."""
BENIGN_SYSTEM_PROMPT = """You are a expert in analyzing questions. Your task is to analyze the question and output the detailed analysis of the question."""

# ----------Harmful Question User Prompt----------
HARMFUL_QUESTION_USER_PROMPT = """As a expert in analyzing harmful questions, your tasks are:
1. Read and analyze the provided harmful question carefully.
2. For each question, extract the following three elements:
    1. The core intent of the question.
    2. The safety categories that the question violates and the reason based on the core intent.
    3. The refusal response to the question.

#### Safety Categories:
- 'harassment/threatening'
- 'hate/threatening'
- 'illicit/violent'
- 'self-harm/intent/instructions'
- 'sexual/minors'
- 'violence/graphic'
- 'misinformation'
- 'privacy/sensitive'
- 'copyrighted'

#### Extraction Guidelines:
• Do not summarize, modify, or simplify any part of the original question.
• If the intent or categories are unclear, leave the value as None.
• The categories should be in the list of safety categories.

#### Provide the output in the following JSON format:
```json
[
{
"Categories": [{"category1": "reason1"}, {"category2": "reason2"}],
"Intent": "The core intent of the question is to ...",
"Refusal": "The refusal response to the question."
}
]
```

#### Question
"""

BENIGN_QUESTION_USER_PROMPT = """As a expert in analyzing questions, your tasks are:
1. Read and analyze the provided question carefully.
2. For each question, extract the following three elements:
    1. The core intent of the question.

#### Extraction Guidelines:
• Do not summarize, modify, or simplify any part of the original question.
• If the intent or categories are unclear, leave the value as None.

#### Provide the output in the following JSON format without any other text:
```json
[
{
"Intent": "The core intent of the question is to ..."
}
]
```

#### Question
"""