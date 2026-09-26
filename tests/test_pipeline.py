from unittest.mock import patch
import pandas as pd

from src.rag.pipeline import ask


@patch("src.rag.pipeline.get_retriever")
def test_ask_returns_fallback_when_no_results(mock_get_retriever):

    mock_retriever = mock_get_retriever.return_value

    mock_retriever.search.return_value = pd.DataFrame()

    answer, results = ask(
        "What are credit card complaints?"
    )

    assert "do not have enough information" in answer
    assert results.empty


@patch("src.rag.pipeline.generate_answer")
@patch("src.rag.pipeline.get_retriever")
def test_ask_returns_answer_when_context_exists(
    mock_get_retriever,
    mock_generate_answer
):

    mock_retriever = mock_get_retriever.return_value

    mock_retriever.search.return_value = pd.DataFrame(
        {
            "text": [
                "Customers reported problems with credit cards."
            ],
            "distance": [
                0.5
            ]
        }
    )

    mock_generate_answer.return_value = (
        "Common complaints are related to credit cards."
    )

    answer, results = ask(
        "What are common credit card complaints?"
    )

    assert answer == (
        "Common complaints are related to credit cards."
    )

    assert len(results) == 1
