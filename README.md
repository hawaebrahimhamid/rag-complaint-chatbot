# CrediTrust Complaint Assistant — RAG System

[![Tests](https://github.com/hawaebrahimhamid/rag-complaint-chatbot/actions/workflows/tests.yml/badge.svg)](https://github.com/hawaebrahimhamid/rag-complaint-chatbot/actions)

An AI-powered **Retrieval-Augmented Generation (RAG)** system that helps financial-service teams analyze customer complaints and retrieve evidence-backed answers from the CFPB Consumer Complaint Database.

## Live Demo

**Streamlit App:** https://creditrust-complaint-chatbot.streamlit.app/

## Overview

Financial institutions receive large volumes of customer complaints, making it difficult to quickly identify recurring issues.

CrediTrust provides a conversational interface that:

- Retrieves relevant complaint evidence using semantic search
- Generates answers using FLAN-T5
- Displays supporting complaint sources
- Rejects out-of-domain questions when relevant evidence is unavailable
- Validates generated answers and falls back to retrieved evidence when generation is unreliable

## Architecture

```text
User Question
      |
      v
Query Embedding
      |
      v
FAISS Similarity Search
      |
      v
Retrieval Relevance Check
      |
      v
Relevant Complaint Evidence
      |
      v
Prompt Construction
      |
      v
FLAN-T5 Generation
      |
      v
Answer Quality Validation
      |
      +----------------+
      |                |
    Valid            Invalid
      |                |
      v                v
Generated       Evidence-Grounded
Answer             Fallback
      |                |
      +-------+--------+
              |
              v
        Answer + Sources
```

## RAG Pipeline

### Retrieval

Complaint narratives are converted into vector embeddings using:

`sentence-transformers/all-MiniLM-L6-v2`

The embeddings are stored in a **FAISS IndexFlatL2** vector index. User questions are embedded and matched against complaint chunks to retrieve relevant evidence.

### Generation

Retrieved complaint evidence is passed to:

`google/flan-t5-small`

The model generates responses using the retrieved context rather than relying only on pretrained knowledge.

### Reliability

The pipeline includes:

- Retrieval relevance filtering
- Answer-quality validation
- Low-quality evidence filtering
- Evidence-grounded fallback responses
- Out-of-domain question handling
- Source display for transparency

## Dataset

The system uses the **CFPB Consumer Complaint Database** and focuses on financial products including:

- Checking / Savings
- Credit Cards
- Money Transfers
- Payday Loans

Complaint narratives are cleaned, chunked, embedded, and indexed for semantic retrieval.

## Tech Stack

| Category        | Technology            |
| --------------- | --------------------- |
| Language        | Python                |
| RAG             | Custom RAG Pipeline   |
| Embeddings      | Sentence Transformers |
| Vector Search   | FAISS                 |
| Generation      | FLAN-T5               |
| UI              | Streamlit             |
| Data Processing | Pandas                |
| Deep Learning   | PyTorch               |
| Testing         | Pytest                |
| CI              | GitHub Actions        |

## Testing

The project includes automated tests for retrieval and pipeline behavior.

```text
7 passed
```

Tests are also executed automatically through GitHub Actions.

## Project Structure

```text
rag-complaint-chatbot/
|
+-- app.py
+-- src/
|   +-- config.py
|   +-- rag/
|       +-- evidence.py
|       +-- generator.py
|       +-- pipeline.py
|       +-- prompt.py
|       +-- retriever.py
|       +-- test_retriever.py
|
+-- tests/
|   +-- test_pipeline.py
|   +-- test_retriever.py
|
+-- notebooks/
+-- vector_store/
+-- docs/
+-- requirements.txt
+-- README.md
```

## Engineering Focus

This project goes beyond a basic RAG implementation by adding reliability mechanisms around retrieval and generation.

The pipeline separates:

**Retrieval -> Evidence Validation -> Generation -> Answer Validation -> Fallback**

This helps reduce unsupported or low-quality responses and makes the system more transparent by showing the complaint evidence used for each answer.

## Current Limitation

The application currently uses **FLAN-T5-small** because of CPU and hardware constraints. While the RAG pipeline and reliability checks are implemented, generated responses can occasionally be incomplete. The system therefore prioritizes retrieved evidence and fallback behavior when generation quality is insufficient.

## Author

**Hawa Ebrahim Hamid**

AI Engineer | Generative AI, RAG & Full-Stack Development

GitHub: https://github.com/hawaebrahimhamid

LinkedIn: https://www.linkedin.com/in/ebrahim-hamid/
