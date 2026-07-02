"""Answer generation (LLM) implementations.

Supports multiple backends through a common interface:
- Ollama (local Llama, Mistral, etc.)
- AWS Bedrock (Claude, Titan)
"""

from confluence_chatbot.generation.base import LLM
from confluence_chatbot.generation.bedrock_llm import BedrockLLM
from confluence_chatbot.generation.ollama_llm import OllamaLLM

__all__ = ["LLM", "OllamaLLM", "BedrockLLM"]
