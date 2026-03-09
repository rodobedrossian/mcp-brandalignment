#!/usr/bin/env python3
"""Assemble report context by combining MCP metrics with Notion semantic-layer notes."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

from notion_semantic_layer import NotionError, search_semantic_layer


def load_metrics_text(path: Optional[Path]) -> str:
    if not path:
        return ""
    if not path.exists():
        raise FileNotFoundError(f"Metrics file not found: {path}")
    return path.read_text(encoding="utf-8").strip()


def build_report_context(topic: str, metrics_text: str) -> str:
    doc = search_semantic_layer(topic)
    sections = [f"# {topic.title()} Report Context"]

    sections.append("## Semantic Layer Notes")
    if doc:
        sections.append(f"**Source**: [{doc.title}]({doc.url})")
        sections.append(doc.markdown or "_Semantic layer document has no extractable content._")
    else:
        sections.append(
            "_No semantic-layer document found. Consider authoring "
            f"a `{topic.title()} - Semantic Layer` page in Notion._"
        )

    sections.append("\n## MCP Metrics Summary")
    sections.append(metrics_text or "_Paste your MySQL MCP findings here._")

    return "\n\n".join(sections).strip()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--topic",
        required=True,
        help="High-level report topic (e.g., 'Restock', 'Buy Box').",
    )
    parser.add_argument(
        "--metrics-file",
        type=Path,
        help="Optional path to a text/markdown file containing MCP query findings.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional output path. Defaults to stdout.",
    )
    args = parser.parse_args()

    metrics_text = load_metrics_text(args.metrics_file)
    try:
        content = build_report_context(args.topic, metrics_text)
    except NotionError as exc:
        parser.error(str(exc))
        return 1

    if args.output:
        args.output.write_text(content + "\n", encoding="utf-8")
    else:
        print(content)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

