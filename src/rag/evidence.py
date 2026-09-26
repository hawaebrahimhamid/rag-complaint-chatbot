import re
from collections import Counter

import pandas as pd


STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "been",
    "but",
    "by",
    "can",
    "could",
    "did",
    "do",
    "does",
    "for",
    "from",
    "had",
    "has",
    "have",
    "how",
    "i",
    "if",
    "in",
    "is",
    "it",
    "me",
    "my",
    "of",
    "on",
    "or",
    "our",
    "that",
    "the",
    "their",
    "them",
    "there",
    "these",
    "this",
    "to",
    "was",
    "were",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "with",
    "you",
    "your",
}


QUESTION_WORDS = {
    "what",
    "why",
    "how",
    "when",
    "where",
    "who",
    "which",
}


def tokenize(text: str) -> list[str]:
    """Convert text into useful lowercase tokens."""

    words = re.findall(
        r"\b[a-zA-Z]{2,}\b",
        text.lower(),
    )

    return [
        word
        for word in words
        if word not in STOPWORDS
    ]


def get_question_type(question: str) -> str:
    """Identify the basic type of question."""

    question = question.lower().strip()

    if question.startswith("why"):
        return "why"

    if question.startswith("how"):
        return "how"

    if question.startswith("when"):
        return "when"

    if question.startswith("where"):
        return "where"

    if question.startswith("who"):
        return "who"

    if question.startswith("which"):
        return "which"

    if question.startswith("what"):
        return "what"

    return "general"


def score_text(question: str, text: str) -> float:
    """
    Score how strongly an evidence piece matches the question.

    The score favors:
    - coverage of important question terms
    - repeated matching terms
    - reasonably informative evidence
    """

    question_tokens = tokenize(question)
    text_tokens = tokenize(text)

    if not question_tokens or not text_tokens:
        return 0.0

    question_set = set(question_tokens)
    text_set = set(text_tokens)

    matched_words = question_set & text_set

    if not matched_words:
        return 0.0

    coverage = len(matched_words) / len(question_set)

    text_counts = Counter(text_tokens)

    repetition = sum(
        min(text_counts[word], 2)
        for word in matched_words
    )

    # Reward evidence containing multiple question terms.
    term_match_score = len(matched_words) * 0.35

    # Slight reward for repeated relevant terms.
    repetition_score = repetition * 0.05

    # Prefer evidence that is not an extremely tiny fragment.
    word_count = len(text.split())

    if word_count < 7:
        length_score = -0.4
    elif word_count < 12:
        length_score = -0.1
    elif word_count <= 45:
        length_score = 0.15
    else:
        length_score = 0.0

    return (
        coverage * 2.0
        + term_match_score
        + repetition_score
        + length_score
    )


def split_into_sentences(text: str) -> list[str]:
    """Split text using normal sentence boundaries."""

    text = re.sub(r"\s+", " ", text).strip()

    if not text:
        return []

    sentences = re.split(
        r"(?<=[.!?])\s+",
        text,
    )

    return [
        sentence.strip()
        for sentence in sentences
        if sentence.strip()
    ]


def split_into_windows(
    text: str,
    window_size: int = 30,
    overlap: int = 10,
) -> list[str]:
    """
    Split punctuation-poor complaint text into larger
    overlapping evidence windows.
    """

    words = text.split()

    if not words:
        return []

    windows = []
    start = 0

    while start < len(words):

        window = words[start:start + window_size]

        if window:
            windows.append(" ".join(window))

        if start + window_size >= len(words):
            break

        start += window_size - overlap

    return windows


def split_into_evidence(text: str) -> list[str]:
    """
    Create evidence pieces.

    Prefer complete sentences. If the complaint has little
    punctuation, use larger overlapping windows.
    """

    text = re.sub(r"\s+", " ", text).strip()

    if not text:
        return []

    sentences = split_into_sentences(text)

    # Use sentences when the text contains real sentence boundaries.
    if len(sentences) > 1:
        return sentences

    # CFPB complaints often contain very little punctuation.
    return split_into_windows(
        text,
        window_size=30,
        overlap=10,
    )


def normalize_text(text: str) -> str:
    """Normalize text for duplicate detection."""

    return re.sub(
        r"\W+",
        " ",
        text.lower(),
    ).strip()


def is_too_similar(
    candidate: str,
    selected: list[str],
) -> bool:
    """Avoid selecting nearly identical evidence."""

    candidate_tokens = set(tokenize(candidate))

    if not candidate_tokens:
        return True

    for existing in selected:

        existing_tokens = set(tokenize(existing))

        if not existing_tokens:
            continue

        intersection = candidate_tokens & existing_tokens

        similarity = (
            len(intersection)
            / min(
                len(candidate_tokens),
                len(existing_tokens),
            )
        )

        if similarity >= 0.80:
            return True

    return False


def clean_evidence(text: str) -> str:
    """Clean an evidence fragment before displaying it."""

    text = re.sub(r"\s+", " ", text).strip()

    # Remove obvious leading/trailing punctuation noise.
    text = text.strip(" ,;:-")

    return text


def build_evidence_answer(
    question: str,
    results: pd.DataFrame,
    max_evidence: int = 2,
) -> str:
    """
    Build a concise answer directly from retrieved complaint evidence.

    This function does not invent facts or summarize beyond the
    retrieved evidence.
    """

    if results.empty:
        return (
            "I do not have enough information from the provided "
            "complaint data."
        )

    evidence_items = []

    for _, row in results.iterrows():

        text = str(
            row.get("text", "")
        ).strip()

        if not text:
            continue

        pieces = split_into_evidence(text)

        for piece in pieces:

            piece = clean_evidence(piece)

            if not piece:
                continue

            word_count = len(piece.split())

            # Avoid tiny fragments that cannot communicate useful evidence.
            if word_count < 7:
                continue

            score = score_text(
                question,
                piece,
            )

            if score <= 0:
                continue

            evidence_items.append(
                {
                    "score": score,
                    "text": piece,
                    "word_count": word_count,
                }
            )

    if not evidence_items:
        return (
            "I do not have enough information from the provided "
            "complaint data."
        )

    # Highest relevance first.
    evidence_items.sort(
        key=lambda item: (
            item["score"],
            min(item["word_count"], 45),
        ),
        reverse=True,
    )

    selected = []
    seen = set()

    for item in evidence_items:

        text = item["text"]

        normalized = normalize_text(text)

        if normalized in seen:
            continue

        if is_too_similar(text, selected):
            continue

        seen.add(normalized)
        selected.append(text)

        if len(selected) >= max_evidence:
            break

    if not selected:
        return (
            "I do not have enough information from the provided "
            "complaint data."
        )

    # Keep the fallback concise.
    evidence_text = " ".join(selected)

    return (
        "Based on the retrieved complaints, customers reported: "
        + evidence_text
    )
