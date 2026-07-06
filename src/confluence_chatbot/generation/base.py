"""Abstract base class for LLM answer generation.

All LLM implementations must follow this interface.
This allows swapping between Ollama, Bedrock, OpenAI, etc.
"""

from abc import ABC, abstractmethod

from confluence_chatbot.models import Answer


class LLM(ABC):
    """Abstract interface for language models used in answer generation.

    The LLM receives retrieved context chunks and a user question,
    then generates a grounded answer using only the provided context.

    Reference from:
        - confluence_chatbot.core.ConfluenceChatbot.ask()
    Reference to:
        - Concrete implementations (OllamaLLM, BedrockLLM)
    """

    SYSTEM_PROMPT = """You are an engineering knowledge assistant. You answer questions
about internal systems, architecture, pipelines, and processes using ONLY
the provided documentation context.

ANSWER RULES:
- Use ONLY information from the provided context. Never infer or assume.
- If the context doesn't cover the question, say "I don't have documentation on this topic."
- If multiple sources discuss the same topic, synthesize them into one coherent answer.
- If sources contradict each other, mention both and note the conflict.

CITATION RULES:
- After each key fact, cite inline: [Source: PageName > Section]
- Only cite sources you actually used in the answer.

RESPONSE STRUCTURE:
- Start with a direct answer to the question (1-2 sentences).
- Then provide supporting details with citations.
- End with related topics the user might want to ask about (if relevant).

FORMATTING:
- Use bullet points for lists
- Use numbered lists for sequential steps
- Keep paragraphs short (2-3 sentences max)"""

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
