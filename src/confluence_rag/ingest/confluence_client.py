"""Confluence API client for fetching pages and attachments.

Handles authentication, pagination, and rate limiting for
both Confluence Cloud and Server instances.
"""

import requests
from atlassian import Confluence
from loguru import logger

from confluence_rag.models import Page


class ConfluenceClient:
    """Client for interacting with the Confluence REST API.

    Provides methods to fetch pages, list spaces, and download
    image attachments.

    Args:
        url: Confluence instance URL (e.g., "https://company.atlassian.net").
        email: User email for authentication.
        token: API token for authentication.
        cloud: Whether this is Confluence Cloud (True) or Server (False).

    Reference from:
        - confluence_rag.core.ConfluenceRAG
    Reference to:
        - atlassian-python-api library
    """

    def __init__(
        self,
        url: str,
        email: str,
        token: str,
        cloud: bool = True,
    ) -> None:
        logger.debug(f"ConfluenceClient.__init__ called with url={url}")
        self._url = url.rstrip("/")
        self._email = email
        self._token = token
        self._cloud = cloud
        self._client = Confluence(
            url=self._url,
            username=self._email,
            password=self._token,
            cloud=self._cloud,
        )

    def fetch_page(self, page_id: str) -> Page:
        """Fetch a single page with its full body content.

        Args:
            page_id: The numeric Confluence page ID.

        Returns:
            Page object with HTML body and metadata.

        Reference from:
            - fetch_space_pages()
            - External callers
        Reference to:
            - Confluence REST API (get_page_by_id)
        """
        logger.debug(f"fetch_page called with page_id={page_id}")
        raw = self._client.get_page_by_id(
            page_id=page_id,
            expand="body.storage,version,space",
        )
        return Page(
            page_id=str(raw["id"]),
            title=raw["title"],
            space_key=raw.get("space", {}).get("key", ""),
            url=f"{self._url}/wiki{raw.get('_links', {}).get('webui', '')}",
            html_body=raw["body"]["storage"]["value"],
            version=raw["version"]["number"],
            last_modified=raw["version"]["when"],
        )

    def fetch_space_pages(self, space_key: str) -> list[Page]:
        """Fetch all pages from a Confluence space.

        Handles pagination automatically for spaces with many pages.

        Args:
            space_key: The space key (e.g., "ENG", "DOCS").

        Returns:
            List of Page objects.

        Reference from:
            - confluence_rag.core.ConfluenceRAG.sync()
        Reference to:
            - Confluence REST API (get_all_pages_from_space)
        """
        logger.debug(f"fetch_space_pages called with space_key={space_key}")
        raw_pages = self._client.get_all_pages_from_space(
            space=space_key,
            expand="body.storage,version",
        )

        pages = []
        for raw in raw_pages:
            pages.append(
                Page(
                    page_id=str(raw["id"]),
                    title=raw["title"],
                    space_key=space_key,
                    url=f"{self._url}/wiki{raw.get('_links', {}).get('webui', '')}",
                    html_body=raw["body"]["storage"]["value"],
                    version=raw["version"]["number"],
                    last_modified=raw["version"]["when"],
                )
            )

        logger.info(f"Fetched {len(pages)} pages from space '{space_key}'")
        return pages

    def fetch_space_page_metadata(self, space_key: str) -> list[dict]:
        """Fetch lightweight metadata for all pages in a space (no body content).

        Used for incremental sync — compare versions without downloading full pages.

        Args:
            space_key: The space key.

        Returns:
            List of dicts with page_id, title, version, last_modified.

        Reference from:
            - ingest.sync_manager.SyncManager
        Reference to:
            - Confluence REST API
        """
        logger.debug(f"fetch_space_page_metadata called with space_key={space_key}")
        raw_pages = self._client.get_all_pages_from_space(
            space=space_key,
            expand="version",
        )

        metadata = []
        for raw in raw_pages:
            metadata.append(
                {
                    "page_id": str(raw["id"]),
                    "title": raw["title"],
                    "version": raw["version"]["number"],
                    "last_modified": raw["version"]["when"],
                }
            )

        return metadata

    def download_attachment(self, page_id: str, filename: str) -> bytes | None:
        """Download an image attachment from a Confluence page.

        Args:
            page_id: The page that owns the attachment.
            filename: The attachment filename.

        Returns:
            Raw file bytes, or None if download failed.

        Reference from:
            - ingest.image_describer.ImageDescriber
        Reference to:
            - Confluence REST API (attachments)
        """
        logger.debug(f"download_attachment called with page_id={page_id}, filename={filename}")

        download_url = f"{self._url}/wiki/rest/api/content/{page_id}/child/attachment"
        response = requests.get(
            download_url,
            auth=(self._email, self._token),
            params={"filename": filename},
        )

        if response.status_code != 200:
            logger.error(f"Failed to get attachment info: {response.status_code}")
            return None

        results = response.json().get("results", [])
        if not results:
            logger.error(f"Attachment not found: {filename}")
            return None

        download_link = results[0]["_links"]["download"]
        full_url = f"{self._url}/wiki{download_link}"

        img_response = requests.get(full_url, auth=(self._email, self._token))
        if img_response.status_code != 200:
            logger.error(f"Failed to download image: {img_response.status_code}")
            return None

        logger.debug(f"Downloaded {filename}: {len(img_response.content)} bytes")
        return img_response.content
