"""
retrieval.py
------------
Given a user's question, find the most relevant chunks of evidence
from the knowledge base we built with ingest.py.

This is "semantic" search: it matches by meaning (via embeddings +
cosine similarity), not just exact keyword overlap.
"""

import chromadb
from sentence_transformers import SentenceTransformer

import config

_embedding_model = None
_collection = None


def _get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = SentenceTransformer(config.EMBEDDING_MODEL_NAME)
    return _embedding_model


def _get_collection():
    global _collection
    if _collection is None:
        client = chromadb.PersistentClient(path=config.CHROMA_DB_DIR)
        _collection = client.get_collection(config.COLLECTION_NAME)
    return _collection


def retrieve_evidence(query: str, top_k: int = None):
    """
    Returns a list of dicts: [{"text": ..., "similarity": ...}, ...]
    sorted by relevance (most relevant first).
    """
    top_k = top_k or config.TOP_K_DOCS
    model = _get_embedding_model()
    collection = _get_collection()

    query_embedding = model.encode([query]).tolist()

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=top_k,
    )

    docs = results["documents"][0]
    distances = results["distances"][0]  # ChromaDB returns distance, lower = more similar

    evidence = []
    for doc, dist in zip(docs, distances):
        # Convert distance to a 0-1 similarity score for readability
        similarity = max(0.0, 1.0 - dist)
        evidence.append({"text": doc, "similarity": similarity})

    return evidence
