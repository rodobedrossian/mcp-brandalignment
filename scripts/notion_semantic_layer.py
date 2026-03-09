"""Utilities for retrieving semantic-layer context from Notion."""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Iterable, List, Optional

import requests

NOTION_API_BASE = "https://api.notion.com/v1"
NOTION_VERSION = os.getenv("NOTION_VERSION", "2022-06-28")


class NotionError(RuntimeError):
    """Raised when Notion API requests fail."""


@dataclass
class SemanticLayerDoc:
    """Metadata and content for a semantic-layer document."""

    title: str
    url: str
    page_id: str
    markdown: str


def _notion_headers() -> dict:
    token = os.getenv("NOTION_API_KEY")
    if not token:
        raise NotionError(
            "NOTION_API_KEY environment variable is required to search Notion."
        )
    return {
        "Authorization": f"Bearer {token}",
        "Notion-Version": NOTION_VERSION,
        "Content-Type": "application/json",
    }


def _rich_text_to_string(rich_text_array: Iterable[dict]) -> str:
    parts: List[str] = []
    for item in rich_text_array:
        text = item.get("plain_text")
        if text:
            parts.append(text)
    return "".join(parts).strip()


def _blocks_to_markdown(blocks: Iterable[dict]) -> str:
    lines: List[str] = []
    for block in blocks:
        block_type = block.get("type")
        if block_type == "paragraph":
            content = _rich_text_to_string(block.get("paragraph", {}).get("rich_text", []))
            if content:
                lines.append(content)
        elif block_type == "heading_1":
            content = _rich_text_to_string(block.get("heading_1", {}).get("rich_text", []))
            if content:
                lines.append(f"# {content}")
        elif block_type == "heading_2":
            content = _rich_text_to_string(block.get("heading_2", {}).get("rich_text", []))
            if content:
                lines.append(f"## {content}")
        elif block_type == "heading_3":
            content = _rich_text_to_string(block.get("heading_3", {}).get("rich_text", []))
            if content:
                lines.append(f"### {content}")
        elif block_type == "bulleted_list_item":
            content = _rich_text_to_string(
                block.get("bulleted_list_item", {}).get("rich_text", [])
            )
            if content:
                lines.append(f"- {content}")
        elif block_type == "numbered_list_item":
            content = _rich_text_to_string(
                block.get("numbered_list_item", {}).get("rich_text", [])
            )
            if content:
                lines.append(f"1. {content}")
        elif block_type == "quote":
            content = _rich_text_to_string(block.get("quote", {}).get("rich_text", []))
            if content:
                lines.append(f"> {content}")
    return "\n".join(lines).strip()


def search_semantic_layer(topic: str) -> Optional[SemanticLayerDoc]:
    """Search Notion for a semantic-layer document matching the topic."""
    headers = _notion_headers()
    query = f"{topic} - Semantic Layer"
    response = requests.post(
        f"{NOTION_API_BASE}/search",
        headers=headers,
        json={
            "query": query,
            "page_size": 5,
            "filter": {"property": "object", "value": "page"},
        },
        timeout=30,
    )
    if response.status_code != 200:
        raise NotionError(
            f"Notion search failed ({response.status_code}): {response.text}"
        )

    results = response.json().get("results", [])
    for result in results:
        if result.get("object") != "page":
            continue
        page_id = result.get("id")
        title_rich = (
            result.get("properties", {})
            .get("title")
            .get("title", [])
            if result.get("properties", {}).get("title")
            else []
        )
        title = _rich_text_to_string(title_rich) or result.get("url", query)
        markdown = fetch_page_markdown(page_id)
        return SemanticLayerDoc(
            title=title,
            url=result.get("url", ""),
            page_id=page_id,
            markdown=markdown,
        )
    return None


def fetch_page_markdown(page_id: str, *, page_size: int = 100) -> str:
    """Fetch page blocks and convert to markdown."""
    headers = _notion_headers()
    response = requests.get(
        f"{NOTION_API_BASE}/blocks/{page_id}/children",
        headers=headers,
        params={"page_size": page_size},
        timeout=30,
    )
    if response.status_code != 200:
        raise NotionError(
            f"Failed to retrieve Notion page blocks ({response.status_code}): {response.text}"
        )
    results = response.json().get("results", [])
    return _blocks_to_markdown(results)

