"""Vision model integration for describing architecture diagrams.

Supports Ollama (local LLaVA) and can be extended for
Claude Vision (Bedrock) or other vision models.
"""

import os
import tempfile

from loguru import logger


class ImageDescriber:
    """Describe images using a vision language model.

    Converts architecture diagrams into detailed text descriptions
    that can be embedded and searched.

    Args:
        model: Ollama model name (e.g., "llava:13b").
        prompt: The instruction prompt sent with the image.

    Reference from:
        - ingest.chunker.Chunker._chunk_images()
    Reference to:
        - Ollama API
    """

    DEFAULT_PROMPT = (
        "Describe this architecture diagram in detail. "
        "Include all components, connections, data flows, protocols, "
        "and any labels or annotations visible. "
        "Be specific about the direction of arrows and relationships."
    )

    def __init__(
        self,
        model: str = "llava:13b",
        prompt: str | None = None,
    ) -> None:
        logger.debug(f"ImageDescriber.__init__ called with model={model}")
        self._model = model
        self._prompt = prompt or self.DEFAULT_PROMPT

    def describe(self, image_bytes: bytes, filename: str = "") -> str | None:
        """Send an image to the vision model and get a text description.

        The image is saved to a temporary file (required by Ollama),
        sent to the model, and the temp file is cleaned up after.

        Args:
            image_bytes: Raw image file content.
            filename: Original filename (for logging/fallback).

        Returns:
            Text description of the image, or None on failure.

        Reference from:
            - Chunker._chunk_images()
        Reference to:
            - Ollama chat API
        """
        logger.debug(f"describe called for filename={filename}")

        try:
            import ollama
        except ImportError:
            logger.error("ollama package not installed. Run: pip install ollama")
            return None

        # Ollama requires a file path, so write to temp file
        suffix = os.path.splitext(filename)[1] if filename else ".png"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(image_bytes)
            tmp_path = tmp.name

        try:
            response = ollama.chat(
                model=self._model,
                messages=[
                    {
                        "role": "user",
                        "content": self._prompt,
                        "images": [tmp_path],
                    }
                ],
            )
            description = response["message"]["content"]
            logger.info(f"Image described: {len(description)} chars for {filename}")
            return description

        except Exception as e:
            logger.error(f"Vision model failed for {filename}: {e}")
            return f"[Architecture diagram: {filename} — description unavailable]"

        finally:
            # Clean up temp file
            os.unlink(tmp_path)
