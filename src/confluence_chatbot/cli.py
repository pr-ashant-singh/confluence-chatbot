"""Command-line interface for confluence-chatbot.

Provides terminal commands to sync Confluence spaces and ask questions.

Usage:
    confluence-chatbot sync --space ENG
    confluence-chatbot sync --space ENG --space DOCS
    confluence-chatbot sync --page-id 1234567890
    confluence-chatbot ask "How does caching work?"
    confluence-chatbot ask  (interactive mode)
"""

import argparse
import os
import sys

from dotenv import load_dotenv
from loguru import logger

from confluence_chatbot import ConfluenceChatbot


def get_rag() -> ConfluenceChatbot:
    """Create a ConfluenceChatbot instance from environment variables.

    Reads configuration from .env file or environment variables:
    - CONFLUENCE_URL (required)
    - CONFLUENCE_EMAIL (required)
    - CONFLUENCE_API_TOKEN (required)
    - VECTOR_STORE (default: "faiss")
    - EMBEDDING_MODEL (default: "BAAI/bge-large-en-v1.5")
    - LLM_MODEL (default: "ollama/llama3.1:8b")
    - ENABLE_IMAGE_DESCRIPTION (default: "false")
    - IMAGE_MODEL (default: "llava:13b")

    Reference from:
        - cmd_sync(), cmd_ask()
    Reference to:
        - ConfluenceChatbot
    """
    load_dotenv()

    confluence_url = os.getenv("CONFLUENCE_URL")
    confluence_email = os.getenv("CONFLUENCE_EMAIL")
    confluence_token = os.getenv("CONFLUENCE_API_TOKEN")

    if not all([confluence_url, confluence_email, confluence_token]):
        print("❌ Missing Confluence credentials.")
        print("   Set CONFLUENCE_URL, CONFLUENCE_EMAIL, CONFLUENCE_API_TOKEN")
        print("   in your .env file or environment variables.")
        sys.exit(1)

    return ConfluenceChatbot(
        confluence_url=confluence_url,
        confluence_email=confluence_email,
        confluence_token=confluence_token,
        vector_store=os.getenv("VECTOR_STORE", "faiss"),
        embedding_model=os.getenv("EMBEDDING_MODEL", "BAAI/bge-large-en-v1.5"),
        llm=os.getenv("LLM_MODEL", "ollama/llama3.1:8b"),
        enable_image_description=os.getenv("ENABLE_IMAGE_DESCRIPTION", "false").lower() == "true",
        image_model=os.getenv("IMAGE_MODEL", "llava:13b"),
    )


def cmd_sync(args):
    """Handle the 'sync' command.

    Reference from:
        - main()
    Reference to:
        - ConfluenceChatbot.sync()
    """
    logger.debug("cmd_sync called")
    rag = get_rag()

    spaces = args.space or []
    page_ids = args.page_id or []

    if not spaces and not page_ids:
        print("❌ Provide at least one --space or --page-id")
        sys.exit(1)

    print("📥 Syncing...")
    if spaces:
        print(f"   Spaces: {', '.join(spaces)}")
    if page_ids:
        print(f"   Pages:  {', '.join(page_ids)}")
    if args.full:
        print("   Mode:   FULL (ignoring cached state)")
    else:
        print("   Mode:   Incremental (only new/changed pages)")
    print()

    stats = rag.sync(spaces=spaces, page_ids=page_ids, full=args.full)

    print("✅ Sync complete!")
    print(f"   Pages processed: {stats['pages_processed']}")
    print(f"   Pages skipped:   {stats.get('pages_skipped', 'N/A')}")
    print(f"   Pages deleted:   {stats.get('pages_deleted', 'N/A')}")
    print(f"   Chunks created:  {stats['chunks_created']}")
    print(f"   Time:            {stats['time_seconds']}s")


def cmd_ask(args):
    """Handle the 'ask' command.

    Reference from:
        - main()
    Reference to:
        - ConfluenceChatbot.ask()
    """
    logger.debug("cmd_ask called")
    rag = get_rag()

    # Get question from args or interactive input
    if args.question:
        question = " ".join(args.question)
    else:
        question = input("\n❓ Ask a question: ")

    if not question.strip():
        print("❌ No question provided")
        sys.exit(1)

    print("\n🔍 Searching knowledge base...")
    answer = rag.ask(question)

    print(f"\n{'=' * 60}")
    print(f"❓ {question}")
    print(f"{'=' * 60}")
    print(f"\n💬 {answer.text}")
    print("\n📚 Sources:")
    for src in answer.sources:
        heading = src.get("heading", "")
        page = src.get("page_title", "")
        ctype = src.get("content_type", "")
        url = src.get("page_url", "")
        print(f"   - [{ctype}] {page} > {heading}")
        if url:
            print(f"     {url}")
    print("=" * 60)


def main():
    """Entry point for the CLI."""
    parser = argparse.ArgumentParser(
        prog="confluence-chatbot",
        description="AI chatbot for Confluence documentation",
    )
    subparsers = parser.add_subparsers(dest="command")

    # sync command
    sync_parser = subparsers.add_parser("sync", help="Ingest Confluence pages")
    sync_parser.add_argument("--space", action="append", help="Confluence space key (can repeat)")
    sync_parser.add_argument("--page-id", action="append", help="Specific page ID (can repeat)")
    sync_parser.add_argument("--full", action="store_true", help="Force full re-sync (ignore cached state)")

    # ask command
    ask_parser = subparsers.add_parser("ask", help="Ask a question")
    ask_parser.add_argument("question", nargs="*", help="The question (or omit for interactive)")

    args = parser.parse_args()

    if args.command == "sync":
        cmd_sync(args)
    elif args.command == "ask":
        cmd_ask(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
