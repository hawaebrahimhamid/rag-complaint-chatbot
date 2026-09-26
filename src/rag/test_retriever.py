from src.rag.retriever import Retriever
from src.config import AppConfig


retriever = Retriever(
    "vector_store/faiss.index",
    "vector_store/metadata.csv"
)


questions = [
    "What problems are customers reporting with checking or savings accounts?",
    "Why are customers complaining about credit cards?",
    "Why was my credit card payment declined?"
]


for question in questions:

    print("\n" + "=" * 70)
    print("QUESTION:", question)
    print("=" * 70)

    embedding = retriever.model.encode([question])

    distances, indices = retriever.index.search(
        embedding,
        AppConfig.TOP_K
    )

    print("\nDistances:", distances[0])

    for i, idx in enumerate(indices[0]):

        print(f"\n--- Result {i + 1} ---")

        text = retriever.metadata.iloc[idx]["text"]

        print(text[:500])
