"""Core orchestrator for confluence-chatbot.

The ConfluenceChatbot class is the main entry point. It wires together
ingestion, embedding, vector storage, retrieval, and generation
into a simple interface.
"""

import time

from loguru import logger

from confluence_chatbot.embedding.base import EmbeddingModel
from confluence_chatbot.embedding.sentence_transformer import SentenceTransformerEmbedding
from confluence_chatbot.generation.base import LLM
from confluence_chatbot.generation.ollama_llm import OllamaLLM
from confluence_chatbot.ingest.chunker import Chunker
from confluence_chatbot.ingest.confluence_client import ConfluenceClient
from confluence_chatbot.ingest.image_describer import ImageDescriber
from confluence_chatbot.ingest.sync_manager import SyncManager
from confluence_chatbot.models import Answer
from confluence_chatbot.vector_store.base import VectorStore
from confluence_chatbot.vector_store.s3_vectors import S3VectorsStore


class ConfluenceChatbot:
    """Main orchestrator for RAG over Confluence documentation.

    Provides a simple interface to:
    - sync(): Ingest Confluence pages into the vector store
    - ask(): Ask questions and get grounded answers

    Args:
        confluence_url: Your Confluence instance URL.
        confluence_email: Authentication email.
        confluence_token: API token for Confluence.
        vector_store: Vector store backend ("s3vectors" or a VectorStore instance).
        embedding_model: Embedding model ("BAAI/bge-large-en-v1.5" or an EmbeddingModel instance).
        llm: LLM for answer generation ("ollama/llama3.1:8b" or an LLM instance).
        s3_bucket_name: S3 Vectors bucket name (required if vector_store="s3vectors").
        s3_index_name: S3 Vectors index name.
        s3_region: AWS region for S3 Vectors.
        s3_profile: AWS profile name for authentication.
        enable_image_description: Whether to describe diagrams with a vision model.
        image_model: Vision model for diagram description.
        top_k: Number of chunks to retrieve per query.

    Usage:
        rag = ConfluenceChatbot(
            confluence_url="https://company.atlassian.net",
            confluence_email="user@company.com",
            confluence_token="...",
            s3_bucket_name="my-vector-bucket",
        )
        rag.sync(spaces=["ENG"])
        answer = rag.ask("How does caching work?")
        print(answer)

    Reference from:
        - User code / CLI
    Reference to:
        - ConfluenceClient, Chunker, EmbeddingModel, VectorStore, LLM
    """

    def __init__(
        self,
        confluence_url: str,
        confluence_email: str,
        confluence_token: str,
        vector_store: str | VectorStore = "s3vectors",
        embedding_model: str | EmbeddingModel = "BAAI/bge-large-en-v1.5",
        llm: str | LLM = "ollama/llama3.1:8b",
        s3_bucket_name: str = "",
        s3_index_name: str = "confluence-docs",
        s3_region: str = "us-east-2",
        s3_profile: str | None = None,
        enable_image_description: bool = False,
        image_model: str = "llava:13b",
        top_k: int = 8,
    ) -> None:
        logger.debug("ConfluenceChatbot.__init__ called")

        # Confluence client
        self._confluence = ConfluenceClient(
            url=confluence_url,
            email=confluence_email,
            token=confluence_token,
        )

        # Embedding model
        if isinstance(embedding_model, str):
            if embedding_model.lower().startswith("bedrock/"):
                from confluence_chatbot.embedding.bedrock import BedrockEmbedding

                self._embedding = BedrockEmbedding(
                    region=s3_region,
                    profile_name=s3_profile,
                )
            else:
                self._embedding = SentenceTransformerEmbedding(model_name=embedding_model)
        else:
            self._embedding = embedding_model

        # Vector store
        if isinstance(vector_store, str) and vector_store.lower() == "s3vectors":
            if not s3_bucket_name:
                raise ValueError("s3_bucket_name is required when using s3vectors store")
            self._vector_store = S3VectorsStore(
                bucket_name=s3_bucket_name,
                index_name=s3_index_name,
                dimension=self._embedding.dimension,
                region=s3_region,
                profile_name=s3_profile,
            )
        elif isinstance(vector_store, str) and vector_store.lower() == "faiss":
            from confluence_chatbot.vector_store.faiss_store import FAISSStore

            self._vector_store = FAISSStore(
                index_path=".data/faiss_index",
                dimension=self._embedding.dimension,
            )
        elif isinstance(vector_store, VectorStore):
            self._vector_store = vector_store
        else:
            raise ValueError(
                f"Unsupported vector_store: {vector_store}. Use 'faiss', 's3vectors', or a VectorStore instance."
            )

        # LLM
        if isinstance(llm, str) and llm.startswith("ollama/"):
            model_name = llm.split("/", 1)[1]
            self._llm = OllamaLLM(model=model_name)
        elif isinstance(llm, str) and llm.startswith("bedrock/"):
            from confluence_chatbot.generation.bedrock_llm import BedrockLLM

            self._llm = BedrockLLM(
                region=s3_region,
                profile_name=s3_profile,
            )
        elif isinstance(llm, LLM):
            self._llm = llm
        else:
            raise ValueError(f"Unsupported llm: {llm}. Use 'ollama/<model>', 'bedrock/claude', or an LLM instance.")

        # Chunker with optional image description
        image_describer = None
        if enable_image_description:
            image_describer = ImageDescriber(model=image_model)

        self._chunker = Chunker(image_describer=image_describer)
        self._top_k = top_k

    def sync(self, spaces: list[str], page_ids: list[str] | None = None, full: bool = False) -> dict:
        """Ingest Confluence pages into the vector store.

        Supports incremental sync — only processes new or changed pages.
        On first run, processes everything. On subsequent runs, compares
        page versions and only re-processes what changed.

        Args:
            spaces: List of Confluence space keys to ingest.
            page_ids: Optional list of specific page IDs to ingest
                (always processes these fully, ignores incremental).
            full: Force a full re-sync (ignore stored state).

        Returns:
            Dict with ingestion stats.

        Reference from:
            - User code
        Reference to:
            - ConfluenceClient, Chunker, EmbeddingModel, VectorStore, SyncManager
        """
        start_time = time.perf_counter()
        logger.debug(f"sync called with spaces={spaces}, page_ids={page_ids}, full={full}")

        # Ensure vector index exists
        if hasattr(self._vector_store, "ensure_index_exists"):
            self._vector_store.ensure_index_exists()

        # Initialize sync manager
        sync_mgr = SyncManager()

        if full:
            sync_mgr.reset()

        # If specific page_ids provided, always process them fully
        if page_ids:
            pages_to_process = []
            for pid in page_ids:
                pages_to_process.append(self._confluence.fetch_page(pid))

            total_chunks = self._process_pages(pages_to_process, sync_mgr)
            elapsed = time.perf_counter() - start_time
            return {
                "pages_processed": len(pages_to_process),
                "pages_skipped": 0,
                "pages_deleted": 0,
                "chunks_created": total_chunks,
                "time_seconds": round(elapsed, 2),
            }

        # Incremental sync for spaces
        total_processed = 0
        total_skipped = 0
        total_deleted = 0
        total_chunks = 0

        for space in spaces:
            # Fetch lightweight metadata (no page content)
            logger.info(f"Checking space '{space}' for changes...")
            page_metadata = self._confluence.fetch_space_page_metadata(space)

            # Determine what changed
            changes = sync_mgr.get_changes(page_metadata, space)

            # Process new and changed pages
            pages_to_fetch = changes["new"] + changes["changed"]
            if pages_to_fetch:
                logger.info(f"Fetching {len(pages_to_fetch)} new/changed pages...")
                pages = []
                for pid in pages_to_fetch:
                    pages.append(self._confluence.fetch_page(pid))

                chunks_created = self._process_pages(pages, sync_mgr)
                total_chunks += chunks_created
                total_processed += len(pages)
            else:
                logger.info(f"No changes in space '{space}'")

            # Delete vectors for removed pages
            for pid in changes["deleted"]:
                prefix = f"page-{pid}-"
                self._vector_store.delete_by_prefix(prefix)
                sync_mgr.mark_deleted(pid)
                total_deleted += 1

            total_skipped += len(changes["unchanged"])

        elapsed = time.perf_counter() - start_time
        stats = {
            "pages_processed": total_processed,
            "pages_skipped": total_skipped,
            "pages_deleted": total_deleted,
            "chunks_created": total_chunks,
            "time_seconds": round(elapsed, 2),
        }

        logger.info(
            f"Sync complete: {stats['pages_processed']} processed, "
            f"{stats['pages_skipped']} unchanged (skipped), "
            f"{stats['pages_deleted']} deleted, "
            f"{stats['chunks_created']} chunks in {stats['time_seconds']}s"
        )
        return stats

    def _process_pages(self, pages: list, sync_mgr: SyncManager) -> int:
        """Process a list of pages: chunk, embed, upload, and track state.

        Args:
            pages: List of Page objects to process.
            sync_mgr: SyncManager to record processed pages.

        Returns:
            Total number of chunks created.

        Reference from:
            - sync()
        Reference to:
            - Chunker, EmbeddingModel, VectorStore, SyncManager
        """
        logger.debug(f"_process_pages called with {len(pages)} pages")
        total_chunks = 0

        for page in pages:
            # Chunk the page
            chunks = self._chunker.chunk_page(page, self._confluence)

            if not chunks:
                # Still track it so we don't re-process next time
                sync_mgr.mark_synced(
                    page_id=page.page_id,
                    title=page.title,
                    version=page.version,
                    last_modified=page.last_modified,
                    chunk_count=0,
                    space_key=page.space_key,
                )
                continue

            # Delete old vectors for this page (in case of re-ingestion)
            prefix = f"page-{page.page_id}-"
            self._vector_store.delete_by_prefix(prefix)

            # Embed all chunks
            texts = [chunk.content for chunk in chunks]
            embeddings = self._embedding.embed_documents(texts)

            # Upload to vector store
            keys = [chunk.vector_key for chunk in chunks]
            metadatas = [chunk.metadata for chunk in chunks]
            self._vector_store.upsert_batch(keys, embeddings, metadatas)

            # Track in sync state
            sync_mgr.mark_synced(
                page_id=page.page_id,
                title=page.title,
                version=page.version,
                last_modified=page.last_modified,
                chunk_count=len(chunks),
                space_key=page.space_key,
            )

            total_chunks += len(chunks)

        return total_chunks

    def ask(self, question: str) -> Answer:
        """Ask a question and get a grounded answer from the knowledge base.

        Full RAG pipeline: embed question → retrieve chunks → generate answer.

        Args:
            question: The question to answer.

        Returns:
            Answer object with generated text and source attribution.

        Reference from:
            - User code
        Reference to:
            - EmbeddingModel, VectorStore, LLM
        """
        start_time = time.perf_counter()
        logger.debug(f"ask called with question={question[:50]}...")

        # Embed the question
        query_vector = self._embedding.embed_query(question)

        # Retrieve relevant chunks
        results = self._vector_store.query(query_vector, top_k=self._top_k)

        if not results:
            return Answer(
                text="No relevant documentation found for this question.",
                question=question,
            )

        # Generate answer from context
        context_chunks = [result.metadata for result in results]
        answer = self._llm.generate(question, context_chunks)

        elapsed = time.perf_counter() - start_time
        logger.info(f"Answer generated in {elapsed:.2f}s")

        return answer
