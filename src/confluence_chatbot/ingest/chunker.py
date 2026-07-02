"""Content-aware document chunker.

Splits Confluence pages into meaningful chunks that respect
document structure: sections, tables, code blocks, and diagrams
each get appropriate treatment.
"""

import re

from loguru import logger

from confluence_chatbot.ingest.html_parser import HTMLParser
from confluence_chatbot.ingest.image_describer import ImageDescriber
from confluence_chatbot.models import Chunk, ContentType, Page


class Chunker:
    """Split Confluence pages into embeddable chunks.

    Handles text, tables, and images differently:
    - Text: split by section headings, respect max size with overlap
    - Tables: keep whole if small, summarize if too large
    - Images: describe with vision model, embed the description

    Args:
        max_chunk_size: Maximum characters per text chunk.
        overlap: Characters of overlap between consecutive chunks.
        image_describer: Optional vision model for describing diagrams.

    Reference from:
        - confluence_chatbot.core.ConfluenceChatbot.sync()
    Reference to:
        - HTMLParser
        - ImageDescriber
    """

    def __init__(
        self,
        max_chunk_size: int = 800,
        overlap: int = 100,
        image_describer: ImageDescriber | None = None,
    ) -> None:
        logger.debug(f"Chunker.__init__ called with max_chunk_size={max_chunk_size}, overlap={overlap}")
        self._max_chunk_size = max_chunk_size
        self._overlap = overlap
        self._parser = HTMLParser()
        self._image_describer = image_describer

    def chunk_page(
        self,
        page: Page,
        confluence_client=None,
    ) -> list[Chunk]:
        """Chunk a Confluence page into embeddable pieces.

        Processes text, tables, and images separately, then combines
        all chunks with appropriate metadata.

        Args:
            page: The Page object to chunk.
            confluence_client: Optional client for downloading images.

        Returns:
            List of Chunk objects ready for embedding.

        Reference from:
            - confluence_chatbot.core.ConfluenceChatbot.sync()
        Reference to:
            - _chunk_text_sections()
            - _chunk_tables()
            - _chunk_images()
        """
        logger.debug(f"chunk_page called for page_id={page.page_id}")

        plain_text = self._parser.extract_plain_text(page.html_body)

        # Skip empty pages
        if len(plain_text.strip()) < 10:
            logger.debug(f"Skipping empty page: {page.title}")
            return []

        # Skip raw data dumps (large pages with no structure)
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(page.html_body, "html.parser")
        html_headings = soup.find_all(["h1", "h2", "h3", "h4"])
        if len(plain_text) > 50000 and len(html_headings) == 0:
            logger.debug(f"Skipping raw data page: {page.title} ({len(plain_text)} chars)")
            return []

        all_chunks = []
        chunk_counter = 0

        # 1. Process images (diagrams)
        if self._image_describer and confluence_client:
            image_chunks = self._chunk_images(page, confluence_client)
            for chunk in image_chunks:
                chunk.chunk_id = f"chunk-{chunk_counter}"
                all_chunks.append(chunk)
                chunk_counter += 1

        # 2. Process tables
        table_chunks = self._chunk_tables(page)
        for chunk in table_chunks:
            chunk.chunk_id = f"chunk-{chunk_counter}"
            all_chunks.append(chunk)
            chunk_counter += 1

        # 3. Process text sections — use HTML headings if available,
        #    fall back to numbered patterns, then paragraph-based splitting
        text_chunks = self._chunk_text_smart(page)
        for chunk in text_chunks:
            chunk.chunk_id = f"chunk-{chunk_counter}"
            all_chunks.append(chunk)
            chunk_counter += 1

        table_count = len([c for c in all_chunks if c.content_type in (ContentType.TABLE, ContentType.TABLE_SUMMARY)])
        logger.info(
            f"Page '{page.title}': {len(all_chunks)} chunks "
            f"(images: {len([c for c in all_chunks if c.content_type == ContentType.DIAGRAM])}, "
            f"tables: {table_count}, "
            f"text: {len([c for c in all_chunks if c.content_type == ContentType.TEXT])})"
        )

        return all_chunks

    def _chunk_text_smart(self, page: Page) -> list[Chunk]:
        """Intelligently chunk text using the best available structure.

        Strategy priority:
        1. If page has HTML headings (h1-h4) → split by headings
        2. If page has numbered headings in text → split by those
        3. Fallback → split by paragraph boundaries respecting max size

        Args:
            page: The source page.

        Returns:
            List of text Chunk objects.

        Reference from:
            - chunk_page()
        Reference to:
            - _chunk_by_html_headings()
            - _chunk_text_sections()
            - _chunk_by_paragraphs()
        """
        logger.debug(f"_chunk_text_smart called for page '{page.title}'")

        from bs4 import BeautifulSoup

        soup = BeautifulSoup(page.html_body, "html.parser")
        html_headings = soup.find_all(["h1", "h2", "h3", "h4"])

        # Strategy 1: HTML headings available
        if html_headings:
            return self._chunk_by_html_headings(page)

        # Strategy 2: Numbered headings in text
        plain_text = self._parser.extract_plain_text(page.html_body)
        numbered_headings = re.findall(r"^\d+\.[\d.]*\s+.+$", plain_text, re.MULTILINE)
        if numbered_headings:
            return self._chunk_text_sections(plain_text, page)

        # Strategy 3: Fallback — split by paragraphs
        return self._chunk_by_paragraphs(plain_text, page)

    def _chunk_by_html_headings(self, page: Page) -> list[Chunk]:
        """Split page content by HTML heading tags (h1, h2, h3, h4).

        Extracts text between consecutive headings as separate chunks.
        This handles pages that use Confluence's visual headings without
        numbered prefixes.

        Args:
            page: The source page.

        Returns:
            List of Chunk objects.

        Reference from:
            - _chunk_text_smart()
        Reference to:
            - BeautifulSoup
        """
        logger.debug("_chunk_by_html_headings called")
        from bs4 import BeautifulSoup, NavigableString

        soup = BeautifulSoup(page.html_body, "html.parser")
        chunks = []

        # Find all heading elements
        headings = soup.find_all(["h1", "h2", "h3", "h4"])

        if not headings:
            return []

        # Collect content between headings
        for i, heading in enumerate(headings):
            heading_text = heading.get_text(strip=True)

            # Gather all content until the next heading
            content_parts = []
            sibling = heading.next_sibling

            while sibling:
                if hasattr(sibling, "name") and sibling.name in ["h1", "h2", "h3", "h4"]:
                    break
                if hasattr(sibling, "get_text"):
                    text = sibling.get_text(separator="\n", strip=True)
                    if text:
                        content_parts.append(text)
                elif isinstance(sibling, NavigableString) and sibling.strip():
                    content_parts.append(sibling.strip())
                sibling = sibling.next_sibling

            content = "\n".join(content_parts).strip()

            # Skip empty sections
            if len(content) < 10:
                continue

            # If section is within limit, keep as one chunk
            if len(content) <= self._max_chunk_size:
                chunks.append(
                    Chunk(
                        chunk_id="",
                        content=content,
                        content_type=ContentType.TEXT,
                        heading=heading_text,
                        page_id=page.page_id,
                        page_title=page.title,
                        page_url=page.url,
                        space_key=page.space_key,
                    )
                )
            else:
                # Split long sections
                sub_chunks = self._split_long_section(content, heading_text, page)
                chunks.extend(sub_chunks)

        # Also capture any content before the first heading
        first_heading = headings[0]
        pre_content_parts = []
        for elem in first_heading.previous_siblings:
            if hasattr(elem, "get_text"):
                text = elem.get_text(separator="\n", strip=True)
                if text:
                    pre_content_parts.append(text)
            elif isinstance(elem, NavigableString) and elem.strip():
                pre_content_parts.append(elem.strip())

        pre_content = "\n".join(reversed(pre_content_parts)).strip()
        if len(pre_content) > 10:
            chunks.insert(
                0,
                Chunk(
                    chunk_id="",
                    content=pre_content,
                    content_type=ContentType.TEXT,
                    heading=f"Introduction - {page.title}",
                    page_id=page.page_id,
                    page_title=page.title,
                    page_url=page.url,
                    space_key=page.space_key,
                ),
            )

        return chunks

    def _chunk_by_paragraphs(self, text: str, page: Page) -> list[Chunk]:
        r"""Fallback chunker: split by paragraph boundaries.

        Used when no headings are detected. Splits text into chunks
        at paragraph breaks (\n\n) while respecting max size.

        Args:
            text: Plain text content.
            page: Source page for metadata.

        Returns:
            List of Chunk objects.

        Reference from:
            - _chunk_text_smart()
        Reference to:
            - None (text processing)
        """
        logger.debug("_chunk_by_paragraphs called")

        # Skip very short pages
        if len(text.strip()) < 30:
            return []

        # If the whole page fits in one chunk, keep it whole
        if len(text) <= self._max_chunk_size:
            return [
                Chunk(
                    chunk_id="",
                    content=text.strip(),
                    content_type=ContentType.TEXT,
                    heading=page.title,
                    page_id=page.page_id,
                    page_title=page.title,
                    page_url=page.url,
                    space_key=page.space_key,
                )
            ]

        # Split by paragraphs (double newline) or single newlines
        paragraphs = text.split("\n\n") if "\n\n" in text else text.split("\n")
        chunks = []
        current_chunk = ""

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            if len(current_chunk) + len(para) + 1 > self._max_chunk_size and current_chunk:
                chunks.append(
                    Chunk(
                        chunk_id="",
                        content=current_chunk.strip(),
                        content_type=ContentType.TEXT,
                        heading=page.title,
                        page_id=page.page_id,
                        page_title=page.title,
                        page_url=page.url,
                        space_key=page.space_key,
                    )
                )
                current_chunk = current_chunk[-self._overlap :] + "\n" + para
            else:
                current_chunk = current_chunk + "\n" + para if current_chunk else para

        if current_chunk.strip() and len(current_chunk.strip()) > 10:
            chunks.append(
                Chunk(
                    chunk_id="",
                    content=current_chunk.strip(),
                    content_type=ContentType.TEXT,
                    heading=page.title,
                    page_id=page.page_id,
                    page_title=page.title,
                    page_url=page.url,
                    space_key=page.space_key,
                )
            )

        return chunks

    def _chunk_text_sections(self, text: str, page: Page) -> list[Chunk]:
        """Split text by section headings, respecting max chunk size.

        Args:
            text: Plain text extracted from the page.
            page: The source page (for metadata).

        Returns:
            List of text Chunk objects.

        Reference from:
            - chunk_page()
        Reference to:
            - _split_into_sections()
        """
        logger.debug("_chunk_text_sections called")
        sections = self._split_into_sections(text)
        chunks = []

        for section in sections:
            content = section["content"]

            # Skip empty or placeholder sections
            if len(content.strip()) <= 5:
                continue

            if len(content) <= self._max_chunk_size:
                chunks.append(
                    Chunk(
                        chunk_id="",  # assigned by caller
                        content=content,
                        content_type=ContentType.TEXT,
                        heading=section["heading"],
                        page_id=page.page_id,
                        page_title=page.title,
                        page_url=page.url,
                        space_key=page.space_key,
                    )
                )
            else:
                # Split long sections by paragraphs
                sub_chunks = self._split_long_section(content, section["heading"], page)
                chunks.extend(sub_chunks)

        return chunks

    def _split_long_section(self, content: str, heading: str, page: Page) -> list[Chunk]:
        """Split a section that exceeds max_chunk_size by paragraphs.

        Uses overlap to preserve context at boundaries.

        Args:
            content: The section text.
            heading: The section heading.
            page: Source page for metadata.

        Returns:
            List of Chunk objects.

        Reference from:
            - _chunk_text_sections()
        Reference to:
            - None (pure text processing)
        """
        paragraphs = content.split("\n\n")
        chunks = []
        current_chunk = ""

        for para in paragraphs:
            if len(current_chunk) + len(para) + 1 > self._max_chunk_size and current_chunk:
                chunks.append(
                    Chunk(
                        chunk_id="",
                        content=current_chunk.strip(),
                        content_type=ContentType.TEXT,
                        heading=heading,
                        page_id=page.page_id,
                        page_title=page.title,
                        page_url=page.url,
                        space_key=page.space_key,
                    )
                )
                current_chunk = current_chunk[-self._overlap :] + "\n" + para
            else:
                current_chunk = current_chunk + "\n" + para if current_chunk else para

        if current_chunk.strip():
            chunks.append(
                Chunk(
                    chunk_id="",
                    content=current_chunk.strip(),
                    content_type=ContentType.TEXT,
                    heading=heading,
                    page_id=page.page_id,
                    page_title=page.title,
                    page_url=page.url,
                    space_key=page.space_key,
                )
            )

        return chunks

    def _split_into_sections(self, text: str) -> list[dict]:
        """Split text by numbered section headings.

        Args:
            text: Full plain text.

        Returns:
            List of dicts with 'heading' and 'content'.

        Reference from:
            - _chunk_text_sections()
        Reference to:
            - None (regex text processing)
        """
        section_pattern = r"^(\d+\.[\d.]*\s+.+)$"
        lines = text.split("\n")
        sections = []
        current_heading = "Introduction"
        current_lines = []

        for line in lines:
            if re.match(section_pattern, line.strip()):
                if current_lines:
                    sections.append(
                        {
                            "heading": current_heading,
                            "content": "\n".join(current_lines).strip(),
                        }
                    )
                current_heading = line.strip()
                current_lines = []
            else:
                current_lines.append(line)

        if current_lines:
            sections.append(
                {
                    "heading": current_heading,
                    "content": "\n".join(current_lines).strip(),
                }
            )

        return sections

    def _chunk_tables(self, page: Page) -> list[Chunk]:
        """Extract and chunk tables from the page HTML.

        Small tables become chunks directly. Large tables get summarized.

        Args:
            page: The source page.

        Returns:
            List of table Chunk objects.

        Reference from:
            - chunk_page()
        Reference to:
            - HTMLParser.extract_tables()
        """
        logger.debug("_chunk_tables called")
        tables = self._parser.extract_tables(page.html_body)
        chunks = []

        for table in tables:
            content = table["content"]
            if not content.strip():
                continue

            if len(content) > self._max_chunk_size:
                summary = self._summarize_table(content, table["heading"])
                chunks.append(
                    Chunk(
                        chunk_id="",
                        content=summary,
                        content_type=ContentType.TABLE_SUMMARY,
                        heading=table["heading"],
                        page_id=page.page_id,
                        page_title=page.title,
                        page_url=page.url,
                        space_key=page.space_key,
                        extra_metadata={"full_table": content},
                    )
                )
            else:
                chunks.append(
                    Chunk(
                        chunk_id="",
                        content=content,
                        content_type=ContentType.TABLE,
                        heading=table["heading"],
                        page_id=page.page_id,
                        page_title=page.title,
                        page_url=page.url,
                        space_key=page.space_key,
                    )
                )

        return chunks

    def _summarize_table(self, table_text: str, heading: str) -> str:
        """Create a concise summary of a large table for embedding.

        Extracts key insights (rows with ✅) for searchability.

        Args:
            table_text: Full table as text.
            heading: Table's section heading.

        Returns:
            Summary string suitable for embedding.

        Reference from:
            - _chunk_tables()
        Reference to:
            - None (text heuristics)
        """
        lines = table_text.split("\n")
        winners = [line for line in lines if "✅" in line]

        if winners:
            summary = f"Comparison table: {heading}. "
            summary += "Key findings: " + "; ".join(winners[:5])
            return summary

        return f"Table: {heading}. " + table_text[:600]

    def _chunk_images(self, page: Page, confluence_client) -> list[Chunk]:
        """Extract, download, and describe images from the page.

        Uses a vision model to convert diagrams into searchable text.

        Args:
            page: The source page.
            confluence_client: Client for downloading image attachments.

        Returns:
            List of diagram Chunk objects.

        Reference from:
            - chunk_page()
        Reference to:
            - HTMLParser.extract_image_filenames()
            - ImageDescriber.describe()
            - ConfluenceClient.download_attachment()
        """
        logger.debug("_chunk_images called")
        if not self._image_describer:
            return []

        filenames = self._parser.extract_image_filenames(page.html_body)
        chunks = []

        for filename in filenames:
            image_bytes = confluence_client.download_attachment(page.page_id, filename)
            if not image_bytes:
                continue

            description = self._image_describer.describe(image_bytes, filename)
            if description:
                chunks.append(
                    Chunk(
                        chunk_id="",
                        content=description,
                        content_type=ContentType.DIAGRAM,
                        heading=f"Diagram: {filename}",
                        page_id=page.page_id,
                        page_title=page.title,
                        page_url=page.url,
                        space_key=page.space_key,
                        extra_metadata={"image_filename": filename},
                    )
                )

        return chunks
