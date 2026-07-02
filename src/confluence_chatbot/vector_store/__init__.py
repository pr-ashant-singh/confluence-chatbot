"""Vector store implementations.

Supports multiple backends through a common interface:
- FAISS (local, free, for development/testing)
- S3 Vectors (AWS managed, production-ready)
"""

from confluence_chatbot.vector_store.base import SearchResult, VectorStore
from confluence_chatbot.vector_store.faiss_store import FAISSStore
from confluence_chatbot.vector_store.s3_vectors import S3VectorsStore

__all__ = ["VectorStore", "SearchResult", "FAISSStore", "S3VectorsStore"]
