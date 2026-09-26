from dataclasses import dataclass

@dataclass(frozen=True)
class AppConfig:
    """Application configuration constants."""

    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    GENERATOR_MODEL: str = "google/flan-t5-small"

    TOP_K: int = 3
    RELEVANCE_THRESHOLD: float = 1.0

    MAX_NEW_TOKENS: int = 100
    MIN_NEW_TOKENS: int = 20

    CHUNK_SIZE: int = 500
    CHUNK_OVERLAP: int = 50

    BATCH_SIZE: int = 2

    VECTOR_STORE_PATH: str = "vector_store"
