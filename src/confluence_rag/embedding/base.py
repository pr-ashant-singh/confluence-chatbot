"""Abstract base class for embedding models.

All embedding implementations must follow this interface.
This ensures the library can swap models without changing pipeline logic.
"""

from abc import ABC, abstractmethod

from loguru import logger


class EmbeddingModel(ABC):
    """Abstract interface for text embedding models.

    Any embedding model (local or API-based) must implement these methods.
    The key contract: documents and queries may use different prefixes,
    but the vector space must be consistent.

    Reference from:
        - confluence_rag.core.ConfluenceRAG
    Reference to:
        - Concrete implementations (SentenceTransformerEmbedding, BedrockEmbedding)
    """

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Return the embedding dimension (e.g., 1024 for bge-large)."""
        ...

    @abstractmethod
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of document texts.

        Documents may receive a model-specific prefix (e.g., "Represent this document: ")
        to optimize for retrieval tasks.

        Args:
            texts: List of document text strings to embed.

        Returns:
            List of embedding vectors, each of length `self.dimension`.
        """
        ...

    @abstractmethod
    def embed_query(self, text: str) -> list[float]:
        """Embed a single query text.

        Queries may receive a different prefix than documents
        (e.g., "Represent this question: ") to optimize retrieval.

        Args:
            text: The query/question text to embed.

        Returns:
            Embedding vector of length `self.dimension`.
        """
        ...

    def embed_documents_batch(self, texts: list[str], batch_size: int = 32) -> list[list[float]]:
        """Embed documents in batches for memory efficiency.

        Default implementation calls embed_documents in chunks.
        Override for backends with native batching.

        Args:
            texts: List of document text strings.
            batch_size: Number of texts to process per batch.

        Returns:
            List of embedding vectors.

        Reference from:
            - ingest pipeline (large document sets)
        Reference to:
            - self.embed_documents()
        """
        logger.debug(f"embed_documents_batch called with {len(texts)} texts, batch_size={batch_size}")
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            embeddings = self.embed_documents(batch)
            all_embeddings.extend(embeddings)
            logger.debug(f"Batch {i // batch_size + 1}: embedded {len(batch)} texts")
        return all_embeddings
