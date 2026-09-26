import faiss
import pandas as pd
from sentence_transformers import SentenceTransformer
from src.config import AppConfig


class Retriever:

    def __init__(
        self,
        index_path: str,
        metadata_path: str
    ) -> None:

        self.index = faiss.read_index(
            index_path
        )

        self.metadata = pd.read_csv(
            metadata_path
        )

        self.model = SentenceTransformer(
             AppConfig.EMBEDDING_MODEL
        )


    def search(
        self,
        question: str,
        k: int = 5
    ) -> pd.DataFrame:

        question_embedding = self.model.encode(
            [question]
        )


        distances, indices = self.index.search(
            question_embedding,
            k
        )


        results = self.metadata.iloc[
             indices[0]
        ].copy()

        results["distance"] = distances[0]

        return results
