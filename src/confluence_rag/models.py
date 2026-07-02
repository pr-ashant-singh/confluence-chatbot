"""Data models for confluence-rag.

These define the core data structures that flow through the pipeline.
All models use dataclasses for clarity and type safety.
"""

from dataclasses import dataclass, field
from enum import Enum


class ContentType(str, Enum):
    """Type of content a chunk was derived from."""

    TEXT = "text"
    TABLE = "table"
    TABLE_SUMMARY = "table_summary"
    DIAGRAM = "diagram"
    CODE = "code"


@dataclass
class Page:
    """Represents a Confluence page with its metadata.

    Reference from:
        - ingest.confluence_client.fetch_page()
        - ingest.confluence_client.fetch_space_pages()
    Reference to:
        - ingest.html_parser.parse_page()
    """

    page_id: str
    title: str
    space_key: str
    url: str
    html_body: str
    version: int
    last_modified: str


@dataclass
class Chunk:
    """A chunk of content ready for embedding.

    Chunks are the atomic units that get embedded and stored in the vector store.
    Each chunk has enough context to be useful on its own when retrieved.

    Reference from:
        - ingest.chunker.chunk_page()
    Reference to:
        - embedding.base.EmbeddingModel.embed()
        - vector_store.base.VectorStore.upsert()
    """

    chunk_id: str
    content: str
    content_type: ContentType
    heading: str
    page_id: str
    page_title: str
    page_url: str
    space_key: str
    embedding: list[float] = field(default_factory=list)
    extra_metadata: dict = field(default_factory=dict)

    @property
    def vector_key(self) -> str:
        """Unique key for this chunk in the vector store."""
        return f"page-{self.page_id}-{self.chunk_id}"

    @property
    def metadata(self) -> dict:
        """Metadata to store alongside the vector in the vector store."""
        return {
            "page_title": self.page_title,
            "page_url": self.page_url,
            "space_key": self.space_key,
            "heading": self.heading,
            "content_type": self.content_type.value,
            "text": self.content[:1000],
        }


@dataclass
class Answer:
    """A generated answer with source attribution.

    Reference from:
        - generation.base.LLM.generate()
    Reference to:
        - User-facing output
    """

    text: str
    question: str
    sources: list[dict] = field(default_factory=list)
    chunks_used: list[Chunk] = field(default_factory=list)

    def __str__(self) -> str:
        """Human-readable answer with sources."""
        source_lines = "\n".join(
            f"  - [{s.get('content_type', '')}] {s.get('heading', '')} ({s.get('page_title', '')})"
            for s in self.sources
        )
        return f"{self.text}\n\nSources:\n{source_lines}"
