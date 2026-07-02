"""Confluence HTML parser.

Parses Confluence's storage format HTML into structured content blocks.
Handles special Confluence macros, tables, code blocks, and image tags.
"""

import re

from bs4 import BeautifulSoup
from loguru import logger


class HTMLParser:
    """Parse Confluence HTML into structured content blocks.

    Confluence uses a custom storage format with special tags like
    <ac:image>, <ac:structured-macro>, <ac:layout>, etc. This parser
    extracts meaningful content while preserving document structure.

    Reference from:
        - ingest.chunker.Chunker
    Reference to:
        - BeautifulSoup
    """

    def extract_plain_text(self, html: str) -> str:
        """Convert Confluence HTML to plain text.

        Strips all HTML tags but preserves text content and
        whitespace structure.

        Args:
            html: Raw Confluence storage format HTML.

        Returns:
            Clean plain text with normalized whitespace.

        Reference from:
            - Chunker.chunk_page()
        Reference to:
            - BeautifulSoup
        """
        logger.debug("extract_plain_text called")
        soup = BeautifulSoup(html, "html.parser")
        return soup.get_text(separator="\n", strip=True)

    def extract_image_filenames(self, html: str) -> list[str]:
        """Find all image attachment filenames in the HTML.

        Confluence stores images as <ac:image> tags with
        <ri:attachment ri:filename="..."> child elements.

        Args:
            html: Raw Confluence HTML.

        Returns:
            List of image filenames found.

        Reference from:
            - Chunker.chunk_page()
        Reference to:
            - regex pattern matching
        """
        logger.debug("extract_image_filenames called")
        pattern = r'ri:filename="([^"]+)"'
        filenames = re.findall(pattern, html)
        logger.debug(f"Found {len(filenames)} images: {filenames}")
        return filenames

    def extract_tables(self, html: str) -> list[dict]:
        """Extract tables from HTML and convert to readable text.

        Each table is returned with its context heading and
        a text representation suitable for embedding.

        Args:
            html: Raw Confluence HTML.

        Returns:
            List of dicts with 'heading' and 'content' keys.

        Reference from:
            - Chunker.chunk_page()
        Reference to:
            - BeautifulSoup
        """
        logger.debug("extract_tables called")
        soup = BeautifulSoup(html, "html.parser")
        tables = soup.find_all("table")
        results = []

        for table in tables:
            heading = "Table"
            prev = table.find_previous(["h1", "h2", "h3", "h4"])
            if prev:
                heading = prev.get_text(strip=True)

            rows = table.find_all("tr")
            table_text_parts = []

            headers = []
            first_row = rows[0] if rows else None
            if first_row:
                headers = [th.get_text(strip=True) for th in first_row.find_all(["th", "td"])]

            for row in rows[1:]:
                cells = [td.get_text(strip=True) for td in row.find_all(["td", "th"])]
                if headers and cells:
                    row_text = " | ".join(f"{h}: {c}" for h, c in zip(headers, cells))
                    table_text_parts.append(row_text)

            table_text = "\n".join(table_text_parts)
            results.append({"heading": heading, "content": table_text})

        return results
