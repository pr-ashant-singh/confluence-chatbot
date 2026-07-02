"""FAISS-based local vector store.

Stores vectors in a FAISS index on disk. Zero cost, no cloud required.
Ideal for development, testing, and small-to-medium datasets (<100K vectors).

FAISS (Facebook AI Similarity Search) provides efficient similarity search
for dense vectors using optimized indexing algorithms.
"""

import json
from pathlib import Path

import numpy as np
from loguru import logger

from confluence_chatbot.vector_store.base import SearchResult, VectorStore


class FAISSStore(VectorStore):
    """Local vector store using FAISS.

    Stores the FAISS index and metadata as files on disk. Supports
    upsert, query, and delete operations with automatic persistence.

    Args:
        index_path: Directory path to store the index files.
        dimension: Vector dimension (must match embedding model output).

    Files created:
        {index_path}/index.faiss    — the FAISS index binary
        {index_path}/metadata.json  — key-to-metadata mapping
        {index_path}/keys.json      — ordered list of vector keys

    Reference from:
        - confluence_chatbot.core.ConfluenceChatbot
    Reference to:
        - faiss library
    """

    def __init__(
        self,
        index_path: str = ".data/faiss_index",
        dimension: int = 1024,
    ) -> None:
        logger.debug(f"FAISSStore.__init__ called with index_path={index_path}, dimension={dimension}")
        self._index_path = Path(index_path)
        self._dimension = dimension
        self._index = None
        self._keys: list[str] = []
        self._metadata: dict[str, dict] = {}

        # Load existing index or create new
        self._load_or_create()

    def _load_or_create(self) -> None:
        """Load an existing index from disk or create a fresh one.

        Reference from:
            - __init__()
        Reference to:
            - faiss.read_index(), faiss.IndexFlatIP()
        """
        try:
            import faiss
        except ImportError:
            raise ImportError("faiss-cpu is required for FAISSStore. Install it with: pip install faiss-cpu")

        index_file = self._index_path / "index.faiss"
        keys_file = self._index_path / "keys.json"
        metadata_file = self._index_path / "metadata.json"

        if index_file.exists() and keys_file.exists() and metadata_file.exists():
            logger.info(f"Loading existing FAISS index from {self._index_path}")
            self._index = faiss.read_index(str(index_file))

            with open(keys_file, "r") as f:
                self._keys = json.load(f)

            with open(metadata_file, "r") as f:
                self._metadata = json.load(f)

            logger.info(f"Loaded {self._index.ntotal} vectors")
        else:
            logger.info(f"Creating new FAISS index (dimension={self._dimension})")
            # Using IndexFlatIP (inner product) with normalized vectors = cosine similarity
            self._index = faiss.IndexFlatIP(self._dimension)
            self._keys = []
            self._metadata = {}

    def _save(self) -> None:
        """Persist the index and metadata to disk.

        Reference from:
            - upsert(), upsert_batch(), delete(), delete_by_prefix()
        Reference to:
            - faiss.write_index()
        """
        import faiss

        self._index_path.mkdir(parents=True, exist_ok=True)

        faiss.write_index(self._index, str(self._index_path / "index.faiss"))

        with open(self._index_path / "keys.json", "w") as f:
            json.dump(self._keys, f)

        with open(self._index_path / "metadata.json", "w") as f:
            json.dump(self._metadata, f)

        logger.debug(f"Saved index with {self._index.ntotal} vectors")

    def upsert(self, key: str, vector: list[float], metadata: dict) -> None:
        """Insert or update a single vector.

        If the key already exists, removes the old vector first.

        Reference from:
            - upsert_batch()
        Reference to:
            - FAISS index operations
        """
        logger.debug(f"upsert called with key={key}")

        # Remove existing if present
        if key in self._keys:
            self._remove_by_key(key)

        # Add new vector
        vec = np.array([vector], dtype=np.float32)
        self._index.add(vec)
        self._keys.append(key)
        self._metadata[key] = metadata

        self._save()

    def upsert_batch(self, keys: list[str], vectors: list[list[float]], metadatas: list[dict]) -> None:
        """Insert or update multiple vectors at once.

        More efficient than calling upsert() in a loop — does a single
        FAISS add and one save operation.

        Reference from:
            - confluence_chatbot.core.ConfluenceChatbot.sync()
        Reference to:
            - FAISS index.add()
        """
        logger.debug(f"upsert_batch called with {len(keys)} vectors")

        # Remove any existing keys first
        existing_keys = [k for k in keys if k in self._keys]
        if existing_keys:
            for k in existing_keys:
                self._remove_by_key(k)

        # Batch add
        vecs = np.array(vectors, dtype=np.float32)
        self._index.add(vecs)
        self._keys.extend(keys)
        for key, meta in zip(keys, metadatas):
            self._metadata[key] = meta

        self._save()
        logger.info(f"Upserted {len(keys)} vectors (total: {self._index.ntotal})")

    def query(
        self,
        vector: list[float],
        top_k: int = 5,
        filter_metadata: dict | None = None,
    ) -> list[SearchResult]:
        """Find the most similar vectors using cosine similarity.

        Args:
            vector: Query vector (normalized).
            top_k: Number of results to return.
            filter_metadata: Optional filter (post-search filtering).

        Returns:
            List of SearchResult objects sorted by similarity.

        Reference from:
            - confluence_chatbot.core.ConfluenceChatbot.ask()
        Reference to:
            - FAISS index.search()
        """
        logger.debug(f"query called with top_k={top_k}")

        if self._index.ntotal == 0:
            logger.warning("Index is empty, no results")
            return []

        # FAISS search
        query_vec = np.array([vector], dtype=np.float32)
        # Request more results if we need to post-filter
        search_k = min(top_k * 3, self._index.ntotal) if filter_metadata else top_k
        distances, indices = self._index.search(query_vec, search_k)

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1:  # FAISS returns -1 for empty slots
                continue

            key = self._keys[idx]
            metadata = self._metadata.get(key, {})

            # Apply metadata filter if specified
            if filter_metadata:
                if not all(metadata.get(k) == v for k, v in filter_metadata.items()):
                    continue

            # Convert inner product distance to cosine distance (1 - similarity)
            # Higher IP = more similar, so distance = 1 - IP
            cosine_distance = 1.0 - float(dist)

            results.append(
                SearchResult(
                    key=key,
                    distance=cosine_distance,
                    metadata=metadata,
                )
            )

            if len(results) >= top_k:
                break

        return results

    def delete(self, key: str) -> None:
        """Delete a single vector by key.

        FAISS doesn't support direct deletion, so we rebuild the index
        without the deleted vector.

        Reference from:
            - delete_by_prefix()
        Reference to:
            - _rebuild_without_keys()
        """
        logger.debug(f"delete called with key={key}")
        if key not in self._keys:
            return
        self._remove_by_key(key)
        self._save()

    def delete_by_prefix(self, prefix: str) -> int:
        """Delete all vectors whose keys start with the given prefix.

        Reference from:
            - confluence_chatbot.core.ConfluenceChatbot.sync()
        Reference to:
            - _rebuild_without_keys()
        """
        logger.debug(f"delete_by_prefix called with prefix={prefix}")
        keys_to_remove = [k for k in self._keys if k.startswith(prefix)]

        if not keys_to_remove:
            return 0

        for key in keys_to_remove:
            self._remove_by_key(key)

        self._save()
        logger.info(f"Deleted {len(keys_to_remove)} vectors with prefix '{prefix}'")
        return len(keys_to_remove)

    def _remove_by_key(self, key: str) -> None:
        """Remove a vector from the index by rebuilding without it.

        FAISS IndexFlatIP doesn't support removal, so we reconstruct
        the vectors array without the target key.

        Reference from:
            - upsert(), delete(), delete_by_prefix()
        Reference to:
            - FAISS index operations, numpy
        """
        import faiss

        if key not in self._keys:
            return

        idx = self._keys.index(key)

        # Reconstruct all vectors
        if self._index.ntotal > 0:
            all_vectors = np.zeros((self._index.ntotal, self._dimension), dtype=np.float32)
            for i in range(self._index.ntotal):
                all_vectors[i] = self._index.reconstruct(i)

            # Remove the target vector
            remaining_vectors = np.delete(all_vectors, idx, axis=0)

            # Rebuild index
            self._index = faiss.IndexFlatIP(self._dimension)
            if len(remaining_vectors) > 0:
                self._index.add(remaining_vectors)

        # Update keys and metadata
        self._keys.pop(idx)
        self._metadata.pop(key, None)
