"""Ollama LLM implementation for local answer generation.

Uses locally-running Ollama models (Llama, Mistral, etc.)
for generating answers from retrieved context.
"""

from loguru import logger

from confluence_chatbot.generation.base import LLM
from confluence_chatbot.models import Answer


class OllamaLLM(LLM):
    """Local LLM via Ollama for answer generation.

    Requires Ollama to be running locally (ollama serve).
    The model must be pulled beforehand (ollama pull llama3.1:8b).

    Args:
        model: Ollama model name (e.g., "llama3.1:8b", "mistral:7b").
        temperature: Sampling temperature (lower = more factual).

    Reference from:
        - confluence_chatbot.core.ConfluenceChatbot.ask()
    Reference to:
        - Ollama chat API
    """

    def __init__(
        self,
        model: str = "llama3.1:8b",
        temperature: float = 0.1,
    ) -> None:
        logger.debug(f"OllamaLLM.__init__ called with model={model}")
        self._model = model
        self._temperature = temperature

    def generate(self, question: str, context_chunks: list[dict]) -> Answer:
        """Generate an answer using Ollama.

        Sends the system prompt, formatted context, and question
        to the local Ollama model.

        Args:
            question: The user's question.
            context_chunks: Retrieved chunk metadata dicts.

        Returns:
            Answer object with generated text and sources.

        Reference from:
            - confluence_chatbot.core.ConfluenceChatbot.ask()
        Reference to:
            - Ollama API (chat)
        """
        logger.debug(f"generate called with question={question[:50]}...")

        try:
            import ollama
        except ImportError:
            logger.error("ollama package not installed. Run: pip install ollama")
            return Answer(
                text="Error: ollama package not installed.",
                question=question,
            )

        context = self._build_context(context_chunks)

        user_message = f"""Context:
{context}

---

Question: {question}

Answer:"""

        response = ollama.chat(
            model=self._model,
            messages=[
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            options={"temperature": self._temperature},
        )

        answer_text = response["message"]["content"]

        sources = [
            {
                "heading": chunk.get("heading", ""),
                "page_title": chunk.get("page_title", ""),
                "content_type": chunk.get("content_type", ""),
                "page_url": chunk.get("page_url", ""),
            }
            for chunk in context_chunks
        ]

        return Answer(
            text=answer_text,
            question=question,
            sources=sources,
        )
