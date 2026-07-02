"""Test: Image description with LLaVA on pages with diagrams.

Tests that architecture diagrams are downloaded from Confluence,
described by LLaVA, and turned into searchable chunks.

Run (make sure ollama serve is running):
    python examples/test_image_description.py
"""

import os

from dotenv import load_dotenv

from confluence_chatbot import ConfluenceChatbot

load_dotenv()


def main():
    # Initialize with image description enabled
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

    # Ingest pages known to have diagrams
    print("📥 Syncing pages with diagrams (LLaVA enabled)...")
    print("   This will be slower — ~10s per diagram\n")

    stats = rag.sync(
        spaces=[],
        page_ids=[os.getenv("TEST_PAGE_ID", "1234567890")],
    )

    print(f"\n✅ Sync complete: {stats}")

    # Ask a question that should be answered by the diagram
    print("\n" + "=" * 60)
    question = "Describe the high-level architecture diagram"
    print(f"❓ {question}")
    print("=" * 60)

    answer = rag.ask(question)
    print(f"\n💬 {answer.text}")
    print(f"\n📚 Sources:")
    for src in answer.sources:
        print(f"   - [{src.get('content_type', '')}] {src.get('heading', '')}")


if __name__ == "__main__":
    main()
