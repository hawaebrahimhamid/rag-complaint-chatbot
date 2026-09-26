from pathlib import Path
from typing import Tuple

import pandas as pd

from src.rag.retriever import Retriever
from src.rag.prompt import PROMPT_TEMPLATE
from src.rag.generator import generate_answer
from src.rag.evidence import build_evidence_answer
from src.config import AppConfig


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def get_retriever() -> Retriever:
    return Retriever(
        str(PROJECT_ROOT / "vector_store" / "faiss.index"),
        str(PROJECT_ROOT / "vector_store" / "metadata.csv")
    )


def is_poor_answer(answer: str, context: str) -> bool:
    """Detect answers that mostly copy the retrieved complaint text."""

    answer_words = set(answer.lower().split())
    context_words = set(context.lower().split())

    if not answer_words:
        return True

    overlap = len(answer_words & context_words) / len(answer_words)

    return overlap > 0.75


def ask(question: str) -> Tuple[str, pd.DataFrame]:

    retriever = get_retriever()

    results = retriever.search(
        question,
        k=AppConfig.TOP_K
    )

    if results.empty or results["distance"].min() > AppConfig.RELEVANCE_THRESHOLD:
        return (
            "I do not have enough information from the provided complaint data.",
            results
        )

    context = "\n".join(
        f"- {text[:300]}"
        for text in results["text"].tolist()
    )

    prompt = PROMPT_TEMPLATE.format(
        context=context,
        question=question
    )

    answer = generate_answer(prompt)

    if is_poor_answer(answer, context):
        answer = build_evidence_answer(
            question,
            results
        )

    return answer, results
