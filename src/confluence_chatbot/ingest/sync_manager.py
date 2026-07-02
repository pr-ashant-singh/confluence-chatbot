"""Incremental sync manager.

Tracks which pages have been ingested and their versions.
On subsequent syncs, only processes pages that are new, changed, or deleted.

State is stored as a JSON file locally (or can be extended to DynamoDB/S3).

How it works:
1. First sync: processes all pages, records their version + last_modified
2. Next sync: fetches page metadata (lightweight), compares against stored state
3. Only fetches full content for pages that changed
4. Deletes vectors for pages that were removed from Confluence
"""

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path

from loguru import logger


@dataclass
class PageSyncState:
    """Tracked state for a single page.

    Reference from:
        - SyncManager
    Reference to:
        - None (data container)
    """

    page_id: str
    title: str
    version: int
    last_modified: str
    chunk_count: int
    space_key: str


class SyncManager:
    """Manages incremental sync state for Confluence pages.

    Stores a record of every ingested page with its version number.
    On the next sync, compares current Confluence state against this
    record to determine what changed.

    Args:
        state_path: Path to the JSON state file.

    Reference from:
        - confluence_chatbot.core.ConfluenceChatbot.sync()
    Reference to:
        - Local filesystem (JSON state file)
    """

    def __init__(self, state_path: str = ".data/sync_state.json") -> None:
        logger.debug(f"SyncManager.__init__ called with state_path={state_path}")
        self._state_path = Path(state_path)
        self._state: dict[str, PageSyncState] = self._load_state()

    def _load_state(self) -> dict[str, PageSyncState]:
        """Load sync state from disk.

        Returns:
            Dict mapping page_id to PageSyncState.

        Reference from:
            - __init__()
        Reference to:
            - filesystem
        """
        if not self._state_path.exists():
            logger.debug("No existing sync state found, starting fresh")
            return {}

        with open(self._state_path, "r") as f:
            raw = json.load(f)

        state = {}
        for page_id, data in raw.items():
            state[page_id] = PageSyncState(**data)

        logger.info(f"Loaded sync state: {len(state)} pages tracked")
        return state

    def _save_state(self) -> None:
        """Persist sync state to disk.

        Reference from:
            - mark_synced(), mark_deleted()
        Reference to:
            - filesystem
        """
        self._state_path.parent.mkdir(parents=True, exist_ok=True)

        raw = {pid: asdict(state) for pid, state in self._state.items()}
        with open(self._state_path, "w") as f:
            json.dump(raw, f, indent=2)

        logger.debug(f"Saved sync state: {len(self._state)} pages")

    def get_changes(self, current_pages: list[dict], space_key: str) -> dict:
        """Compare current Confluence pages against stored state.

        Determines which pages are new, changed, or deleted.

        Args:
            current_pages: List of page metadata dicts from Confluence
                (must have page_id, title, version, last_modified).
            space_key: The space being synced.

        Returns:
            Dict with keys:
                - 'new': list of page_ids (never seen before)
                - 'changed': list of page_ids (version increased)
                - 'deleted': list of page_ids (in state but not in Confluence)
                - 'unchanged': list of page_ids (same version)

        Reference from:
            - confluence_chatbot.core.ConfluenceChatbot.sync()
        Reference to:
            - self._state
        """
        logger.debug(f"get_changes called for space={space_key} with {len(current_pages)} pages")

        current_ids = set()
        new_pages = []
        changed_pages = []
        unchanged_pages = []

        for page_meta in current_pages:
            pid = page_meta["page_id"]
            current_ids.add(pid)

            if pid not in self._state:
                new_pages.append(pid)
            elif page_meta["version"] > self._state[pid].version:
                changed_pages.append(pid)
            else:
                unchanged_pages.append(pid)

        # Find deleted pages (in our state but no longer in Confluence for this space)
        space_pages_in_state = {pid for pid, s in self._state.items() if s.space_key == space_key}
        deleted_pages = list(space_pages_in_state - current_ids)

        logger.info(
            f"Sync diff for space '{space_key}': "
            f"{len(new_pages)} new, {len(changed_pages)} changed, "
            f"{len(deleted_pages)} deleted, {len(unchanged_pages)} unchanged"
        )

        return {
            "new": new_pages,
            "changed": changed_pages,
            "deleted": deleted_pages,
            "unchanged": unchanged_pages,
        }

    def mark_synced(
        self,
        page_id: str,
        title: str,
        version: int,
        last_modified: str,
        chunk_count: int,
        space_key: str,
    ) -> None:
        """Record that a page has been successfully synced.

        Call this after a page is ingested and its vectors uploaded.

        Args:
            page_id: The page ID.
            title: Page title.
            version: Current page version number.
            last_modified: ISO timestamp of last modification.
            chunk_count: Number of chunks created from this page.
            space_key: The space this page belongs to.

        Reference from:
            - confluence_chatbot.core.ConfluenceChatbot.sync()
        Reference to:
            - self._state, self._save_state()
        """
        logger.debug(f"mark_synced called for page_id={page_id}, version={version}")
        self._state[page_id] = PageSyncState(
            page_id=page_id,
            title=title,
            version=version,
            last_modified=last_modified,
            chunk_count=chunk_count,
            space_key=space_key,
        )
        self._save_state()

    def mark_deleted(self, page_id: str) -> None:
        """Remove a page from sync state (it was deleted from Confluence).

        Args:
            page_id: The page ID to remove.

        Reference from:
            - confluence_chatbot.core.ConfluenceChatbot.sync()
        Reference to:
            - self._state, self._save_state()
        """
        logger.debug(f"mark_deleted called for page_id={page_id}")
        if page_id in self._state:
            del self._state[page_id]
            self._save_state()

    def get_page_state(self, page_id: str) -> PageSyncState | None:
        """Get the sync state for a specific page.

        Args:
            page_id: The page ID.

        Returns:
            PageSyncState or None if not tracked.
        """
        return self._state.get(page_id)

    @property
    def total_pages_tracked(self) -> int:
        """Number of pages currently in sync state."""
        return len(self._state)

    def reset(self) -> None:
        """Clear all sync state. Forces a full re-sync next time.

        Reference from:
            - CLI (confluence-chatbot sync --full)
        Reference to:
            - filesystem
        """
        logger.info("Resetting sync state")
        self._state = {}
        if self._state_path.exists():
            os.remove(self._state_path)
