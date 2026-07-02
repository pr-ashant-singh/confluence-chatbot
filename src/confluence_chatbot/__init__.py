"""confluence-chatbot: Plug-and-play RAG for Confluence documentation.

A library that turns your Confluence docs into a searchable knowledge base
with support for text, tables, and architecture diagrams.

Usage:
    from confluence_chatbot import ConfluenceChatbot

    rag = ConfluenceChatbot(
        confluence_url="https://company.atlassian.net",
        confluence_email="user@company.com",
        confluence_token="...",
        vector_store="s3vectors",
        embedding_model="BAAI/bge-large-en-v1.5",
        llm="ollama/llama3.1:8b",
    )

    rag.sync(spaces=["ENG"])
    answer = rag.ask("How does caching work?")
"""

from confluence_chatbot.core import ConfluenceChatbot
from confluence_chatbot.models import Answer, Chunk, Page

__version__ = "0.1.0"
__all__ = ["ConfluenceChatbot", "Answer", "Chunk", "Page"]
