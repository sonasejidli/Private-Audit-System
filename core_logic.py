**Private Audit System - Core RAG Pipeline**
----------------------------------------
This module defines the Advanced RAG architecture comprising:
1. Hybrid Search (Vector + Keyword)
2. Cross-Encoder Reranking
3. Contextual Compression
"""

from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

def build_advanced_rag_engine(splits):
    """
    Constructs the Hybrid Search + Reranking pipeline.

    Args:
        splits (list): List of chunked documents.

    Returns:
        retriever: The final ContextualCompressionRetriever engine.
    """

    # 1. Dense Retrieval (Semantic Search)
    # Captures the "meaning" of the query.
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
    vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 20}) # Broad recall

    # 2. Sparse Retrieval (Keyword Search)
    # Captures exact keyword matches (IDs, Codes, Specific Terms).
    bm25_retriever = BM25Retriever.from_documents(splits)
    bm25_retriever.k = 20 # Broad recall

    # 3. Hybrid Fusion (Ensemble)
    # Combines both results using Reciprocal Rank Fusion (RRF).
    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, vector_retriever],
        weights=[0.5, 0.5] # Balanced weight
    )

    # 4. Cross-Encoder Reranking (The Filter)
    # Re-scores the top 20 documents to find the true relevance.
    # Uses a specialized model trained on MS-MARCO.
    model = HuggingFaceCrossEncoder(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")
    compressor = CrossEncoderReranker(model=model, top_n=5) # Select top 5 strictly relevant docs

    # 5. Final Pipeline Construction
    final_retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=ensemble_retriever
    )

    return final_retriever
