# Private-Audit-System
Privacy-First, High Precision, and Hybrid Retrieval Technology
##  Project Overview
The **Private Audit System** is an advanced **Retrieval-Augmented Generation (RAG)** solution designed to analyze sensitive financial documents (PDFs) in a strictly **local (offline)** environment. The primary objective is to eliminate **Data Leakage** risks while overcoming the limitations of traditional vector search.

Unlike basic RAG systems that rely solely on semantic similarity, this system employs a multi-stage pipeline to understand both the semantic meaning and the specific context of complex audit reports.

---

##  Architecture & Algorithmic Approach

This project implements a **Multi-Stage Retrieval Pipeline** rather than a standard "naive" RAG approach:

### 1. Hybrid Search (Recall Maximization)
Single-mode retrieval is often insufficient for technical domains. We utilize an **Ensemble Retriever** combining two distinct algorithms:
* **Dense Retrieval (Semantic):** Uses **`intfloat/multilingual-e5-small`** embeddings. This state-of-the-art multilingual model is specifically chosen for its superior performance in processing **Azerbaijani** and other non-English texts, ensuring accurate semantic capture compared to standard English-centric models.
* **Sparse Retrieval (Keyword):** Uses the `BM25` (Best Matching 25) algorithm to capture exact keyword matches, crucial for specific IDs, codes, and terminologies.
* **Weighted Fusion:** Results are combined using **Reciprocal Rank Fusion (RRF)** with a balanced weight distribution (0.5/0.5).

### 2. Cross-Encoder Reranking (Precision Maximization)
The top candidates (Top-20) from the retrieval stage are passed through a "Reranker":
* **Model:** `Cross-Encoder/MS-MARCO-MiniLM-L-6-v2`.
* **Logic:** Unlike Bi-Encoders, this model processes the Query and Document **simultaneously** via a full self-attention mechanism. It assigns a relevance score to each document, effectively filtering out "noise" and hallucinations before the data reaches the LLM.

### 3. Generative Inference (Local LLM)
* **Model:** `Gemma 2 (9B)` - Google's high-performance open-weights model.
* **Deployment:** Orchestrated via `Ollama` for fully local CPU/GPU inference. No data is ever transmitted to the cloud.

---

##  Tech Stack

| Component | Technology / Library | Purpose |
| :--- | :--- | :--- |
| **LLM** | `Ollama` + `Gemma 2` | Local Intelligence & Inference |
| **Orchestration** | `LangChain` | RAG Pipeline Management |
| **Vector DB** | `ChromaDB` | Vector Storage & Indexing |
| **Embeddings** | `Sentence-Transformers` | Text-to-Vector Conversion |`multilingual-e5-small` | High-performance Multilingual Vectorization |
| **Reranker** | `Cross-Encoder` (HuggingFace) | Contextual Filtering & Re-ranking |
| **Interface** | `Streamlit` | User Interface (UI) |

---

##  Installation & Setup

### 1. Prerequisites
* Python 3.10+
* Ollama (Must be installed on the host machine)
* GPU (Recommended for faster inference, but CPU compatible)

### 2. Clone the Repository
```bash
git clone [https://github.com/sonasejidli/Private-Audit-System.git](https://github.com/sonasejidli/Private-Audit-System.git)
cd private-audit-system

### 3. Initialize Ollama Server
ollama serve
ollama pull gemma2

### 4. Run the Application
streamlit run app.py

---

### 5: `requirements.txt`
*Use this version to ensure anyone cloning your repo gets the exact correct versions of the libraries.*

```text
# Core Framework
langchain==0.3.0
langchain-community==0.3.0
langchain-core==0.3.0

# UI & Web
streamlit==1.38.0

# LLM & Local Serving
ollama==0.3.3

# Vector Database & Embeddings
chromadb==0.5.5
sentence-transformers==3.0.1

# Advanced Retrieval Algorithms
rank_bm25==0.2.2          # For Sparse Search (BM25)
torch                     # For Cross-Encoder & Embedding models

# PDF Processing
pypdf==4.3.1

# Utilities
python-dotenv
requests


