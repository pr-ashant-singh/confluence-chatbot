"""Embedding model using sentence-transformers (local, open-source).

Supports any model from HuggingFace that works with the sentence-transformers library.
Recommended: BAAI/bge-large-en-v1.5 for English retrieval tasks.
"""

from loguru import logger
from sentence_transformers import SentenceTransformer

from confluence_chatbot.embedding.base import EmbeddingModel


class SentenceTransformerEmbedding(EmbeddingModel):
    """Local embedding model using sentence-transformers.

    Runs entirely on your machine — no API calls, no costs.
    The model is downloaded once from HuggingFace and cached locally.

    Args:
        model_name: HuggingFace model identifier.
            Recommended: "BAAI/bge-large-en-v1.5" (1024 dims, high quality)
            Alternative: "all-MiniLM-L6-v2" (384 dims, faster, lighter)
        document_prefix: Prefix added to documents before embedding.
        query_prefix: Prefix added to queries before embedding.

    Reference from:
        - confluence_chatbot.core.ConfluenceChatbot
    Reference to:
        - sentence_transformers.SentenceTransformer
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-large-en-v1.5",
        document_prefix: str = "Represent this document: ",
        query_prefix: str = "Represent this question: ",
    ) -> None:
        logger.debug(f"SentenceTransformerEmbedding.__init__ called with model_name={model_name}")
        self._model_name = model_name
        self._document_prefix = document_prefix
        self._query_prefix = query_prefix
        self._model: SentenceTransformer | None = None

    @property
    def model(self) -> SentenceTransformer:
        """Lazy-load the model on first use.

        This avoids loading the large model (~1.3GB) at import time.
        The model stays in memory after first load for fast subsequent calls.
        """
        if self._model is None:
            logger.info(f"Loading embedding model: {self._model_name}")
            self._model = SentenceTransformer(self._model_name)
            logger.info("Embedding model loaded successfully")
        return self._model

    @property
    def dimension(self) -> int:
        """Return the embedding dimension from the loaded model."""
        return self.model.get_embedding_dimension()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed document texts with document prefix.

        Args:
            texts: List of document text strings.

        Returns:
            List of normalized embedding vectors.

        Reference from:
            - Ingestion pipeline
        Reference to:
            - SentenceTransformer.encode()
        """
        logger.debug(f"embed_documents called with {len(texts)} texts")
        prefixed_texts = [self._document_prefix + text for text in texts]
        embeddings = self.model.encode(
            prefixed_texts,
            normalize_embeddings=True,
            show_progress_bar=len(texts) > 10,
            batch_size=8,
        )
        return embeddings.tolist()

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query with query prefix.

        Args:
            text: The question/query text.

        Returns:
            Normalized embedding vector.

        Reference from:
            - Retrieval pipeline
        Reference to:
            - SentenceTransformer.encode()
        """
        logger.debug(f"embed_query called with text={text[:50]}...")
        prefixed_text = self._query_prefix + text
        embedding = self.model.encode(
            prefixed_text,
            normalize_embeddings=True,
        )
        return embedding.tolist()
