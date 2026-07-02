"""Quick test of the confluence-chatbot library — full local pipeline.

This tests:
1. Connect to Confluence and fetch a page
2. Chunk it (text + tables + diagrams)
3. Embed with BAAI
4. Store in FAISS (local)
5. Query and get an answer via Ollama/Llama

Prerequisites:
- .env file with CONFLUENCE_URL, CONFLUENCE_EMAIL, CONFLUENCE_API_TOKEN
- Ollama running (ollama serve) with llama3.1:8b pulled
- pip install -e ".[all]"

Run:
    python examples/quickstart.py
"""

import os

from dotenv import load_dotenv

from confluence_chatbot import ConfluenceChatbot

load_dotenv()


def main():
    # Initialize with all-local setup
    rag = ConfluenceChatbot(
        confluence_url=os.getenv("CONFLUENCE_URL"),
        confluence_email=os.getenv("CONFLUENCE_EMAIL"),
        confluence_token=os.getenv("CONFLUENCE_API_TOKEN"),
        vector_store="faiss",
        embedding_model="BAAI/bge-large-en-v1.5",
        llm="ollama/llama3.1:8b",
        enable_image_description=True,
        image_model="llava:13b",
    )

    # Sync a single page
    print("📥 Syncing page...")
    stats = rag.sync(
        spaces=[],
        page_ids=[os.getenv("TEST_PAGE_ID", "1234567890")],
    )
    print(f"✅ Sync complete: {stats}")

    # Ask questions
    questions = [
        "How does caching work?",
        "What are the main architecture decisions?",
        "What is the request flow?",
    ]

    for question in questions:
        print(f"\n{'=' * 60}")
        print(f"❓ {question}")
        print("=" * 60)
        answer = rag.ask(question)
        print(f"\n💬 {answer.text}")
        print(f"\n📚 Sources:")
        for src in answer.sources:
            print(f"   - [{src.get('content_type', '')}] {src.get('heading', '')}")


if __name__ == "__main__":
    main()
