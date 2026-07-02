"""Bedrock Titan Embeddings implementation.

Uses Amazon Bedrock's Titan Text Embeddings V2 model via API.
No local model required — just a boto3 call. Fast, managed, low cost.

Titan Embeddings V2 outputs 1024-dimensional vectors by default.
"""

import json

import boto3
from loguru import logger

from confluence_chatbot.embedding.base import EmbeddingModel


class BedrockEmbedding(EmbeddingModel):
    """Embedding model using Amazon Bedrock Titan Text Embeddings V2.

    Makes API calls to Bedrock — no local model, no PyTorch, lightweight.
    Requires AWS credentials with bedrock:InvokeModel permission.

    Args:
        model_id: Bedrock model identifier.
        region: AWS region where Bedrock is available.
        profile_name: AWS CLI profile for authentication.
        dimension: Output dimension (Titan v2 supports 256, 512, 1024).

    Reference from:
        - confluence_chatbot.core.ConfluenceChatbot
    Reference to:
        - Bedrock Runtime API (invoke_model)
    """

    def __init__(
        self,
        model_id: str = "amazon.titan-embed-text-v2:0",
        region: str = "us-east-2",
        profile_name: str | None = None,
        dimension: int = 1024,
    ) -> None:
        logger.debug(f"BedrockEmbedding.__init__ called with model_id={model_id}, region={region}")
        self._model_id = model_id
        self._dimension = dimension
        self._client = self._create_client(region, profile_name)

    def _create_client(self, region: str, profile_name: str | None):
        """Create the Bedrock Runtime client.

        Reference from:
            - __init__()
        Reference to:
            - boto3.Session
        """
        session_kwargs = {"region_name": region}
        if profile_name:
            session_kwargs["profile_name"] = profile_name
        session = boto3.Session(**session_kwargs)
        return session.client("bedrock-runtime")

    @property
    def dimension(self) -> int:
        """Return the configured embedding dimension."""
        return self._dimension

    def _invoke_model(self, text: str) -> list[float]:
        """Call Bedrock Titan to embed a single text.

        Args:
            text: The text to embed.

        Returns:
            Embedding vector.

        Reference from:
            - embed_documents(), embed_query()
        Reference to:
            - Bedrock Runtime API
        """
        body = json.dumps(
            {
                "inputText": text,
                "dimensions": self._dimension,
                "normalize": True,
            }
        )

        response = self._client.invoke_model(
            modelId=self._model_id,
            body=body,
        )

        response_body = json.loads(response["body"].read())
        return response_body["embedding"]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed document texts using Bedrock Titan.

        Note: Titan doesn't use prefix instructions like BGE.
        Each text is sent as-is.

        Args:
            texts: List of document text strings.

        Returns:
            List of embedding vectors.

        Reference from:
            - Ingestion pipeline
        Reference to:
            - _invoke_model()
        """
        logger.debug(f"embed_documents called with {len(texts)} texts")
        embeddings = []
        for i, text in enumerate(texts):
            # Titan has a 8192 token limit — truncate if needed
            truncated = text[:20000]  # rough char limit
            embedding = self._invoke_model(truncated)
            embeddings.append(embedding)

            if (i + 1) % 50 == 0:
                logger.debug(f"Embedded {i + 1}/{len(texts)} documents")

        logger.debug(f"Embedded all {len(texts)} documents")
        return embeddings

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query text using Bedrock Titan.

        Args:
            text: The query/question text.

        Returns:
            Embedding vector.

        Reference from:
            - Retrieval pipeline
        Reference to:
            - _invoke_model()
        """
        logger.debug(f"embed_query called with text={text[:50]}...")
        return self._invoke_model(text)
