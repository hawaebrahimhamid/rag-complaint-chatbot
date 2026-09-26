# CrediTrust Complaint Assistant — RAG System

[![Tests](https://github.com/hawaebrahimhamid/rag-complaint-chatbot/actions/workflows/tests.yml/badge.svg)](https://github.com/hawaebrahimhamid/rag-complaint-chatbot/actions)

An AI-powered **Retrieval-Augmented Generation (RAG)** system that helps financial-service teams quickly understand customer complaints.

The system retrieves relevant complaints from the CFPB dataset, generates an answer using **FLAN-T5**, and provides supporting complaint sources for transparency.

## 🚀 Live Demo

**Streamlit App:** https://creditrust-complaint-chatbot.streamlit.app/

## 🎯 Problem

Financial institutions receive large volumes of customer complaints. Manually reviewing these complaints makes it difficult to quickly identify recurring problems and understand what customers are experiencing.

CrediTrust Complaint Assistant provides a conversational interface for querying complaint data and retrieving evidence-backed responses.

## ✨ Key Features

* 🔎 **Semantic search** over customer complaints using FAISS
* 🤖 **RAG-based question answering** with FLAN-T5
* 📚 **Source transparency** — retrieved complaints are shown with answers
* 🛡️ **Retrieval relevance filtering** to reduce unrelated results
* ✅ **Answer-quality validation** to detect weak generated responses
* 🧾 **Evidence-grounded fallback** when generation is unreliable
* 🚫 **Out-of-domain rejection** when the complaint dataset cannot answer a question
* 🧪 Automated testing with **pytest**
* 🔄 Continuous testing with **GitHub Actions**
* 📊 Streamlit dashboard for exploring complaint data

## 🏗️ Architecture

```text
User Question
      │
      ▼
Query Embedding
      │
      ▼
FAISS Similarity Search
      │
      ▼
Retrieval Relevance Check
      │
      ▼
Relevant Complaint Evidence
      │
      ▼
Prompt Construction
      │
      ▼
FLAN-T5 Generation
      │
      ▼
Answer Quality Validation
      │
      ├───────────────┐
      │               │
   Valid           Invalid
      │               │
      ▼               ▼
Generated       Evidence-Grounded
Answer              Fallback
      │               │
      └───────┬───────┘
              ▼
        Answer + Sources
```

## 🧠 Technical Approach

### Retrieval

Customer complaint narratives are converted into vector embeddings using:

**`sentence-transformers/all-MiniLM-L6-v2`**

The embeddings are stored in a **FAISS IndexFlatL2** vector index. For each question, the system retrieves the most relevant complaint chunks.

### Generation

The retrieved complaint evidence is provided to:

**`google/flan-t5-small`**

The model generates an answer based on the retrieved complaint context rather than relying only on its pretrained knowledge.

### Reliability Improvements

The project includes several safeguards beyond a basic RAG pipeline:

* Retrieval relevance threshold
* Answer-quality checks
* Duplicate/low-quality evidence filtering
* Evidence-grounded fallback responses
* Out-of-domain question handling
* Source display for explainability

These improvements were added as part of the Week 12 production improvements.

## 📊 Dataset

The project uses the **Consumer Financial Protection Bureau (CFPB) Consumer Complaint Database**.

The pipeline processes complaint narratives and focuses on financial product categories including:

* Checking / Savings
* Credit Cards
* Money Transfers
* Payday Loans

Complaint narratives are cleaned, chunked, embedded, and indexed for semantic retrieval.

## 🛠️ Tech Stack

| Category      | Technology            |
| ------------- | --------------------- |
| Language      | Python                |
| RAG           | Custom RAG pipeline   |
| Embeddings    | Sentence Transformers |
| Vector Search | FAISS                 |
| Generation    | FLAN-T5               |
| UI            | Streamlit             |
| Testing       | Pytest                |
| CI/CD         | GitHub Actions        |
| Data          | Pandas                |
| Deep Learning | PyTorch               |

## 🧪 Testing

The project includes automated tests covering retrieval and pipeline behavior.

Current test status:

```text
7 passed
```

Tests are also executed automatically through GitHub Actions.

## 📸 Interface

The Streamlit application provides:

* A conversational complaint search interface
* Generated answers
* Retrieved complaint sources
* Complaint-category statistics
* Visual exploration of the processed dataset

## 📁 Project Structure

```text
rag-complaint-chatbot/
│
├── app.py
├── src/
│   ├── config.py
│   └── rag/
│       ├── evidence.py
│       ├── generator.py
│       ├── pipeline.py
│       ├── prompt.py
│       ├── retriever.py
│       └── test_retriever.py
│
├── tests/
│   ├── test_pipeline.py
│   └── test_retriever.py
│
├── notebooks/
│   ├── task1_eda_preprocessing.ipynb
│   └── task2_chunking_embedding.ipynb
│
├── vector_store/
├── docs/
├── requirements.txt
└── README.md
```

## 🚀 Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/hawaebrahimhamid/rag-complaint-chatbot.git
cd rag-complaint-chatbot
```

### 2. Create and activate a virtual environment

```bash
python -m venv venv
```

Windows:

```powershell
.\venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the application

```bash
streamlit run app.py
```

The application will open in your browser.

### 5. Run tests

```bash
pytest -q
```

## ⚠️ Limitations

* The current generator is **FLAN-T5-small**, selected with CPU/hardware constraints in mind.
* Complaint narratives contain noisy and incomplete text.
* Generated responses can occasionally be incomplete, which is why evidence-based fallback and answer-quality checks are included.
* The system is designed for complaint-data exploration rather than general-purpose question answering.

## 🔮 Future Improvements

* Upgrade to a larger instruction-tuned generation model
* Improve retrieval ranking with reranking
* Add multilingual support
* Add conversation history
* Add authentication and user management
* Improve evaluation with a larger set of RAG quality metrics

## 👩‍💻 Author

**Hawa Ebrahim Hamid**

AI Engineer | Generative AI, RAG & Full-Stack Development

GitHub: https://github.com/hawaebrahimhamid

LinkedIn: https://www.linkedin.com/in/hawa-ebrahim-hamid/
