"""Abstract base class for vector stores.

All vector store implementations must follow this interface.
This allows swapping between S3 Vectors, FAISS, pgvector, etc.
without changing pipeline logic.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class SearchResult:
    """A single result from a vector similarity search.

    Reference from:
        - VectorStore.query()
    Reference to:
        - retrieval.retriever.Retriever
    """

    key: str
    distance: float
    metadata: dict


class VectorStore(ABC):
    """Abstract interface for vector storage and retrieval.

    Vector stores are responsible for:
    - Storing vectors with associated metadata
    - Performing similarity search (k-nearest neighbors)
    - Managing vector lifecycle (upsert, delete)

    Reference from:
        - confluence_rag.core.ConfluenceRAG
    Reference to:
        - Concrete implementations (S3VectorsStore, FAISSStore)
    """

    @abstractmethod
    def upsert(self, key: str, vector: list[float], metadata: dict) -> None:
        """Insert or update a single vector.

        If a vector with the same key exists, it is overwritten.

        Args:
            key: Unique identifier for this vector.
            vector: The embedding vector (list of floats).
            metadata: Key-value pairs to store alongside the vector.
        """
        ...

    @abstractmethod
    def upsert_batch(self, keys: list[str], vectors: list[list[float]], metadatas: list[dict]) -> None:
        """Insert or update multiple vectors in a batch.

        More efficient than calling upsert() in a loop.

        Args:
            keys: List of unique identifiers.
            vectors: List of embedding vectors.
            metadatas: List of metadata dicts (one per vector).
        """
        ...

    @abstractmethod
    def query(
        self,
        vector: list[float],
        top_k: int = 5,
        filter_metadata: dict | None = None,
    ) -> list[SearchResult]:
        """Find the most similar vectors to the query vector.

        Args:
            vector: The query embedding vector.
            top_k: Number of results to return.
            filter_metadata: Optional metadata filter to narrow results.

        Returns:
            List of SearchResult objects ordered by similarity (closest first).
        """
        ...

    @abstractmethod
    def delete(self, key: str) -> None:
        """Delete a single vector by key.

        Args:
            key: The unique identifier of the vector to delete.
        """
        ...

    @abstractmethod
    def delete_by_prefix(self, prefix: str) -> int:
        """Delete all vectors whose keys start with the given prefix.

        Useful for removing all chunks from a specific page.

        Args:
            prefix: The key prefix to match (e.g., "page-12345-").

        Returns:
            Number of vectors deleted.
        """
        ...
