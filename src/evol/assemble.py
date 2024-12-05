"""
Assemble the dataset by evolving the original dataset.
The steps are:
1. Sample a question from the question pool.
2. Sample a question strategy.
3. Enrich the question template by the question strategy.
4. Generate the question by the enriched template.
5. Modify the answer with the answer reasoning.
6. Save the evolved question and answer.
"""
# TODO: assemble the question and answer into a jsonl file -> save in the data/evolved_data folder

from src.evol.questions import QuestionStrategy
from src.evol.answers import answer_format

def evol_question(question):
    question_strategy = QuestionStrategy()

    strategy = question_strategy.sample_strategy()
    question_instance = question_strategy.enrich_question_with_strategy(question, strategy)
    print("Question instance: ", question_instance)

    return question_instance

def evol_answer(question,question_category):
    answer_instance = answer_format.format(question=question, category=question_category)
    print("Answer instance: ", answer_instance)
    return answer_instance

def main():
    question = "What is the capital of France?"
    question_category = "commen knowledge"
    question_instance = evol_question(question)
    answer_instance = evol_answer(question, question_category)

if __name__ == "__main__":
    main()