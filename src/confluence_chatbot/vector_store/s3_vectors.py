"""S3 Vectors implementation of the vector store interface.

Amazon S3 Vectors is a purpose-built vector storage service that provides
sub-second similarity search at scale. This implementation handles
index creation, vector upload, and querying.
"""

import boto3
from loguru import logger

from confluence_chatbot.vector_store.base import SearchResult, VectorStore


class S3VectorsStore(VectorStore):
    """Vector store backed by Amazon S3 Vectors.

    Manages a vector index within an S3 vector bucket. Handles index creation,
    batch uploads, querying, and deletion.

    Args:
        bucket_name: Name of the S3 vector bucket.
        index_name: Name of the vector index within the bucket.
        dimension: Vector dimension (must match embedding model output).
        region: AWS region where the bucket lives.
        profile_name: AWS CLI profile name for authentication.

    Reference from:
        - confluence_chatbot.core.ConfluenceChatbot
    Reference to:
        - boto3 s3vectors client
    """

    def __init__(
        self,
        bucket_name: str,
        index_name: str,
        dimension: int = 1024,
        region: str = "us-east-2",
        profile_name: str | None = None,
    ) -> None:
        logger.debug(f"S3VectorsStore.__init__ called with bucket={bucket_name}, index={index_name}")
        self._bucket_name = bucket_name
        self._index_name = index_name
        self._dimension = dimension
        self._region = region
        self._profile_name = profile_name
        self._client = self._create_client()

    def _create_client(self):
        """Create the boto3 s3vectors client.

        Reference from:
            - __init__()
        Reference to:
            - boto3.Session
        """
        logger.debug("_create_client called")
        session_kwargs = {"region_name": self._region}
        if self._profile_name:
            session_kwargs["profile_name"] = self._profile_name
        session = boto3.Session(**session_kwargs)
        return session.client("s3vectors")

    def ensure_index_exists(self) -> None:
        """Create the vector index if it doesn't already exist.

        Idempotent — safe to call multiple times. Checks for existence first.

        Reference from:
            - confluence_chatbot.core.ConfluenceChatbot.sync()
        Reference to:
            - S3 Vectors API (list_indexes, create_index)
        """
        logger.debug("ensure_index_exists called")
        try:
            response = self._client.list_indexes(vectorBucketName=self._bucket_name)
            existing_names = [idx["indexName"] for idx in response.get("indexes", [])]
            if self._index_name in existing_names:
                logger.info(f"Index '{self._index_name}' already exists")
                return
        except Exception as e:
            logger.warning(f"Could not list indexes: {e}. Attempting creation.")

        logger.info(f"Creating index '{self._index_name}' (dimension={self._dimension}, metric=cosine)")
        self._client.create_index(
            vectorBucketName=self._bucket_name,
            indexName=self._index_name,
            dataType="float32",
            dimension=self._dimension,
            distanceMetric="cosine",
        )
        logger.info(f"Index '{self._index_name}' created successfully")

    def upsert(self, key: str, vector: list[float], metadata: dict) -> None:
        """Insert or update a single vector.

        Reference from:
            - upsert_batch() (internal)
        Reference to:
            - S3 Vectors API (put_vectors)
        """
        logger.debug(f"upsert called with key={key}")
        self._client.put_vectors(
            vectorBucketName=self._bucket_name,
            indexName=self._index_name,
            vectors=[
                {
                    "key": key,
                    "data": {"float32": vector},
                    "metadata": metadata,
                }
            ],
        )

    def upsert_batch(self, keys: list[str], vectors: list[list[float]], metadatas: list[dict]) -> None:
        """Insert or update multiple vectors in batches.

        S3 Vectors supports batch writes for efficiency. This method
        handles batching automatically.

        Reference from:
            - confluence_chatbot.core.ConfluenceChatbot.sync()
        Reference to:
            - S3 Vectors API (put_vectors)
        """
        logger.debug(f"upsert_batch called with {len(keys)} vectors")
        batch_size = 10

        for i in range(0, len(keys), batch_size):
            batch_vectors = [
                {
                    "key": keys[j],
                    "data": {"float32": vectors[j]},
                    "metadata": metadatas[j],
                }
                for j in range(i, min(i + batch_size, len(keys)))
            ]

            self._client.put_vectors(
                vectorBucketName=self._bucket_name,
                indexName=self._index_name,
                vectors=batch_vectors,
            )
            logger.debug(f"Uploaded batch {i // batch_size + 1} ({len(batch_vectors)} vectors)")

        logger.info(f"Uploaded {len(keys)} vectors total")

    def query(
        self,
        vector: list[float],
        top_k: int = 5,
        filter_metadata: dict | None = None,
    ) -> list[SearchResult]:
        """Find the most similar vectors.

        Reference from:
            - retrieval.retriever.Retriever.retrieve()
        Reference to:
            - S3 Vectors API (query_vectors)
        """
        logger.debug(f"query called with top_k={top_k}")

        query_params = {
            "vectorBucketName": self._bucket_name,
            "indexName": self._index_name,
            "queryVector": {"float32": vector},
            "topK": top_k,
            "returnMetadata": True,
            "returnDistance": True,
        }

        if filter_metadata:
            query_params["filter"] = filter_metadata

        response = self._client.query_vectors(**query_params)

        results = []
        for v in response.get("vectors", []):
            results.append(
                SearchResult(
                    key=v.get("key", ""),
                    distance=v.get("distance", 0.0),
                    metadata=v.get("metadata", {}),
                )
            )

        return results

    def delete(self, key: str) -> None:
        """Delete a single vector by key.

        Reference from:
            - delete_by_prefix()
        Reference to:
            - S3 Vectors API (delete_vectors)
        """
        logger.debug(f"delete called with key={key}")
        self._client.delete_vectors(
            vectorBucketName=self._bucket_name,
            indexName=self._index_name,
            keys=[key],
        )

    def delete_by_prefix(self, prefix: str) -> int:
        """Delete all vectors whose keys start with the given prefix.

        Lists vectors matching the prefix, then deletes them in batch.

        Reference from:
            - sync_manager (when a page is re-ingested or deleted)
        Reference to:
            - S3 Vectors API (list_vectors, delete_vectors)
        """
        logger.debug(f"delete_by_prefix called with prefix={prefix}")
        # List vectors with the prefix
        response = self._client.list_vectors(
            vectorBucketName=self._bucket_name,
            indexName=self._index_name,
            segmentCount=1,
            segmentIndex=0,
        )

        keys_to_delete = [v["key"] for v in response.get("vectors", []) if v["key"].startswith(prefix)]

        if not keys_to_delete:
            return 0

        # Delete in batches
        batch_size = 10
        for i in range(0, len(keys_to_delete), batch_size):
            batch = keys_to_delete[i : i + batch_size]
            self._client.delete_vectors(
                vectorBucketName=self._bucket_name,
                indexName=self._index_name,
                keys=batch,
            )

        logger.info(f"Deleted {len(keys_to_delete)} vectors with prefix '{prefix}'")
        return len(keys_to_delete)
