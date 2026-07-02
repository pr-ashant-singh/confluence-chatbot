"""Debug: Check why some pages produce 0 chunks.

Fetches all pages from a space and shows their content
structure to understand why the chunker misses them.

Run:
    python examples/debug_chunks.py
"""

import os

from dotenv import load_dotenv

from confluence_rag.ingest.confluence_client import ConfluenceClient
from confluence_rag.ingest.html_parser import HTMLParser

load_dotenv()


def main():
    client = ConfluenceClient(
        url=os.getenv("CONFLUENCE_URL"),
        email=os.getenv("CONFLUENCE_EMAIL"),
        token=os.getenv("CONFLUENCE_API_TOKEN"),
    )

    parser = HTMLParser()

    space_key = os.getenv("TEST_SPACE_KEY", "TEAM")
    pages = client.fetch_space_pages(space_key)
    print(f"Total pages: {len(pages)}\n")

    for page in pages:
        plain_text = parser.extract_plain_text(page.html_body)
        text_len = len(plain_text.strip())

        # Check for numbered headings (current chunker pattern)
        import re
        numbered_headings = re.findall(r"^\d+\.[\d.]*\s+.+$", plain_text, re.MULTILINE)

        # Check for HTML headings in raw HTML
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(page.html_body, "html.parser")
        html_headings = soup.find_all(["h1", "h2", "h3", "h4"])
        heading_texts = [h.get_text(strip=True) for h in html_headings]

        # Flag pages that would produce 0 chunks
        has_numbered = len(numbered_headings) > 0
        has_html_headings = len(heading_texts) > 0
        is_empty = text_len <= 5

        status = "✅" if has_numbered else ("⚠️" if has_html_headings else "❌")

        if is_empty:
            status = "🔴 EMPTY"

        print(f"{status} {page.title}")
        print(f"     Text length: {text_len} chars")
        print(f"     Numbered headings: {len(numbered_headings)}")
        print(f"     HTML headings: {heading_texts[:5]}")

        # Show first 100 chars for pages with 0 numbered headings
        if not has_numbered and not is_empty:
            preview = plain_text[:150].replace("\n", " | ")
            print(f"     Preview: {preview}")
        print()


if __name__ == "__main__":
    main()
