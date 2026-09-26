import pandas as pd
from unittest.mock import MagicMock, patch

from src.rag.retriever import Retriever


@patch("src.rag.retriever.SentenceTransformer")
@patch("src.rag.retriever.pd.read_csv")
@patch("src.rag.retriever.faiss.read_index")
def test_retriever_initialization(
    mock_read_index,
    mock_read_csv,
    mock_sentence_transformer
):
    mock_read_index.return_value = MagicMock()
    mock_read_csv.return_value = pd.DataFrame({"text": ["sample"]})
    mock_sentence_transformer.return_value = MagicMock()

    retriever = Retriever(
        "fake.index",
        "fake.csv"
    )

    assert retriever is not None


@patch("src.rag.retriever.SentenceTransformer")
@patch("src.rag.retriever.pd.read_csv")
@patch("src.rag.retriever.faiss.read_index")
def test_search_returns_dataframe(
    mock_read_index,
    mock_read_csv,
    mock_sentence_transformer,
):
    mock_index = MagicMock()
    mock_index.search.return_value = (
    [[0.5]],
    [[0]]
)

    mock_read_index.return_value = mock_index

    mock_read_csv.return_value = pd.DataFrame(
        {
            "text": ["Complaint example"]
        }
    )

    mock_model = MagicMock()
    mock_model.encode.return_value = [[0.1, 0.2]]

    mock_sentence_transformer.return_value = mock_model

    retriever = Retriever(
        "fake.index",
        "fake.csv"
    )

    results = retriever.search("credit card")

    assert isinstance(results, pd.DataFrame)
    assert len(results) == 1
