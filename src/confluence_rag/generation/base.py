"""Abstract base class for LLM answer generation.

All LLM implementations must follow this interface.
This allows swapping between Ollama, Bedrock, OpenAI, etc.
"""

from abc import ABC, abstractmethod

from loguru import logger

from confluence_rag.models import Answer


class LLM(ABC):
    """Abstract interface for language models used in answer generation.

    The LLM receives retrieved context chunks and a user question,
    then generates a grounded answer using only the provided context.

    Reference from:
        - confluence_rag.core.ConfluenceRAG.ask()
    Reference to:
        - Concrete implementations (OllamaLLM, BedrockLLM)
    """

    SYSTEM_PROMPT = """You are an engineering knowledge assistant.
Your job is to answer questions about engineering systems, architecture, and pipelines.

RULES:
- Answer ONLY based on the provided context below.
- If the context does not contain enough information to answer, say "I don't have enough information to answer this based on the available documentation."
- Cite which section the information comes from.
- Be concise and technical.
- Do not make up information."""

    @abstractmethod
    def generate(self, question: str, context_chunks: list[dict]) -> Answer:
        """Generate an answer from retrieved context.

        Args:
            question: The user's question.
            context_chunks: List of dicts with 'heading', 'text',
                'page_title', 'content_type' keys.

        Returns:
            Answer object with generated text and source attribution.
        """
        ...

    def _build_context(self, context_chunks: list[dict]) -> str:
        """Format retrieved chunks into a context string for the LLM.

        Args:
            context_chunks: List of chunk metadata dicts.

        Returns:
            Formatted context string.

        Reference from:
            - generate() implementations
        Reference to:
            - None (string formatting)
        """
        parts = []
        for chunk in context_chunks:
            source = chunk.get("page_title", "Unknown")
            heading = chunk.get("heading", "")
            text = chunk.get("text", "")
            parts.append(f"[Source: {source} > {heading}]\n{text}")

        return "\n\n---\n\n".join(parts)
