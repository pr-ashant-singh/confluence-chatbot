"""Embedding model implementations.

Supports multiple backends through a common interface:
- sentence-transformers (BAAI/bge-large, MiniLM, etc.)
- AWS Bedrock (Titan Embeddings v2)
"""

from confluence_rag.embedding.base import EmbeddingModel
from confluence_rag.embedding.bedrock import BedrockEmbedding
from confluence_rag.embedding.sentence_transformer import SentenceTransformerEmbedding

__all__ = ["EmbeddingModel", "SentenceTransformerEmbedding", "BedrockEmbedding"]
