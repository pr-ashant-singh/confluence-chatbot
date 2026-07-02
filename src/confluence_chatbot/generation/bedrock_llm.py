"""Bedrock Claude LLM implementation for answer generation.

Uses Amazon Bedrock's Claude model via API for high-quality,
grounded answer generation from retrieved context.
"""

import json

import boto3
from loguru import logger

from confluence_chatbot.generation.base import LLM
from confluence_chatbot.models import Answer


class BedrockLLM(LLM):
    """Answer generation using Claude via Amazon Bedrock.

    Provides high-quality, technically accurate answers with strong
    instruction following (grounding, citation, "I don't know").

    Args:
        model_id: Bedrock model identifier for Claude.
        region: AWS region.
        profile_name: AWS CLI profile for authentication.
        max_tokens: Maximum tokens in the generated answer.
        temperature: Sampling temperature (lower = more factual).

    Reference from:
        - confluence_chatbot.core.ConfluenceChatbot.ask()
    Reference to:
        - Bedrock Runtime API (invoke_model)
    """

    def __init__(
        self,
        model_id: str = "us.anthropic.claude-haiku-4-5-20251001-v1:0",
        region: str = "us-east-2",
        profile_name: str | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.1,
    ) -> None:
        logger.debug(f"BedrockLLM.__init__ called with model_id={model_id}, region={region}")
        self._model_id = model_id
        self._max_tokens = max_tokens
        self._temperature = temperature
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

    def generate(self, question: str, context_chunks: list[dict]) -> Answer:
        """Generate an answer using Claude via Bedrock.

        Sends the system prompt, formatted context, and question
        to Claude for grounded answer generation.

        Args:
            question: The user's question.
            context_chunks: Retrieved chunk metadata dicts.

        Returns:
            Answer object with generated text and sources.

        Reference from:
            - confluence_chatbot.core.ConfluenceChatbot.ask()
        Reference to:
            - Bedrock Runtime API (invoke_model with Messages API)
        """
        logger.debug(f"generate called with question={question[:50]}...")

        context = self._build_context(context_chunks)

        user_message = f"""Context:
{context}

---

Question: {question}

Answer:"""

        body = json.dumps(
            {
                "anthropic_version": "bedrock-2023-05-31",
                "system": self.SYSTEM_PROMPT,
                "messages": [
                    {"role": "user", "content": user_message},
                ],
                "max_tokens": self._max_tokens,
                "temperature": self._temperature,
            }
        )

        response = self._client.invoke_model(
            modelId=self._model_id,
            body=body,
        )

        response_body = json.loads(response["body"].read())
        answer_text = response_body["content"][0]["text"]

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
