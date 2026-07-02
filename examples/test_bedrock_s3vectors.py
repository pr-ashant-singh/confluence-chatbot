"""Test: Full AWS production setup — Bedrock Titan + Claude + S3 Vectors.

This tests the production-grade pipeline:
- Embeddings: Bedrock Titan v2 (API, no local model)
- Vector Store: S3 Vectors (managed, scalable)
- LLM: Claude 3.5 Sonnet via Bedrock (high quality)

Prerequisites:
- AWS profile with Bedrock + S3 Vectors access
- Bedrock model access enabled for Titan Embed + Claude
- S3 Vector bucket exists

Run:
    python examples/test_bedrock_s3vectors.py
"""

import os

from dotenv import load_dotenv

from confluence_rag import ConfluenceRAG

load_dotenv()

AWS_PROFILE = os.getenv("AWS_PROFILE", "your-aws-profile")
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
S3_BUCKET = os.getenv("S3_BUCKET_NAME", "your-s3-bucket")
S3_INDEX = os.getenv("S3_INDEX_NAME", "confluence-docs-bedrock")


def main():
    print("🚀 Testing Bedrock + S3 Vectors production setup\n")

    rag = ConfluenceRAG(
        confluence_url=os.getenv("CONFLUENCE_URL"),
        confluence_email=os.getenv("CONFLUENCE_EMAIL"),
        confluence_token=os.getenv("CONFLUENCE_API_TOKEN"),
        vector_store="s3vectors",
        embedding_model="bedrock/titan",
        llm="bedrock/claude",
        s3_bucket_name=S3_BUCKET,
        s3_index_name=S3_INDEX,
        s3_region=AWS_REGION,
        s3_profile=AWS_PROFILE,
        enable_image_description=False,  # use Claude Vision later
    )

    # Sync one page first
    print("📥 Syncing one page with Bedrock Titan embeddings...")
    stats = rag.sync(
        spaces=[],
        page_ids=[os.getenv("TEST_PAGE_ID", "1234567890")],
    )
    print(f"✅ Sync: {stats}\n")

    # Ask questions
    questions = [
        "How does caching work?",
        "What are the main architecture decisions?",
    ]

    for question in questions:
        print(f"{'=' * 60}")
        print(f"❓ {question}")
        answer = rag.ask(question)
        print(f"💬 {answer.text[:300]}...")
        print(f"📚 Sources: {[s.get('heading', '')[:30] for s in answer.sources]}")
        print()

    print("✅ Bedrock + S3 Vectors pipeline working!")


if __name__ == "__main__":
    main()
