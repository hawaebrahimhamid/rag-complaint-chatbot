PROMPT_TEMPLATE = """
Answer the question using only the complaint information below.

COMPLAINTS:
{context}

QUESTION:
{question}

Write one short sentence answering the question.
State only the main problem described in the complaints.
Use simple, direct language.
Do not copy the complaints word-for-word.
Do not add information that is not in the complaints.

ANSWER:
"""
