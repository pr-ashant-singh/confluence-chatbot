"""Test: Ingest a full Confluence space and query across all pages.

This tests:
- Fetching all pages from a space
- Chunking multiple pages
- Embedding everything
- Querying across the full corpus

Run:
    python examples/test_full_space.py
"""

import os
import time

from dotenv import load_dotenv

from confluence_chatbot import ConfluenceChatbot

load_dotenv()


def main():
    start_time = time.perf_counter()

    # Initialize — using FAISS locally, no image description for speed
    rag = ConfluenceChatbot(
        confluence_url=os.getenv("CONFLUENCE_URL"),
        confluence_email=os.getenv("CONFLUENCE_EMAIL"),
        confluence_token=os.getenv("CONFLUENCE_API_TOKEN"),
        vector_store="faiss",
        embedding_model="BAAI/bge-large-en-v1.5",
        llm="ollama/llama3.1:8b",
        enable_image_description=False,  # Skip diagrams for speed on first test
    )

    # Sync a full space
    space_key = os.getenv("TEST_SPACE_KEY", "TEAM")
    print(f"📥 Syncing full {space_key} space...")
    print("   (This may take a few minutes depending on how many pages exist)\n")

    stats = rag.sync(spaces=[space_key])

    print(f"\n✅ Sync complete!")
    print(f"   Pages processed: {stats['pages_processed']}")
    print(f"   Chunks created:  {stats['chunks_created']}")
    print(f"   Time:            {stats['time_seconds']}s")

    # Ask a few questions across the full corpus
    questions = [
        "How does caching work?",
        "What AWS services are used?",
        "How is data synced between storage layers?",
        "What is the overall architecture?",
    ]

    print(f"\n{'=' * 60}")
    print("QUERYING ACROSS FULL SPACE")
    print("=" * 60)

    for question in questions:
        print(f"\n❓ {question}")
        print("-" * 40)
        answer = rag.ask(question)
        # Show just first 200 chars of answer for overview
        print(f"💬 {answer.text[:200]}...")
        print(f"📚 Sources: {[s.get('heading', '')[:30] for s in answer.sources]}")

    elapsed = time.perf_counter() - start_time
    print(f"\n{'=' * 60}")
    print(f"Total time: {elapsed:.1f}s")
    print("=" * 60)


if __name__ == "__main__":
    main()
