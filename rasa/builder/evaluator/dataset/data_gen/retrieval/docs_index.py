"""Parse the documentation repo into a structured page index.

Walks the docs/ repo for .mdx files, extracts frontmatter and content,
and builds a DocIndex that maps each page to its URL, title, and cleaned
markdown body. Final output is in jsonl format.

Usage:
    python -m rasa.builder.evaluator.dataset.data_gen.retrieval.docs_index \
        --docs-repo path-to-docs-repo \
        --output path-to-output-dir/docs_index.jsonl
"""

import argparse
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Tuple

import structlog
from pydantic import BaseModel, Field

from rasa.builder.evaluator.dataset.data_gen.retrieval.constants import (
    DOCS_EXCLUDED_DIRS,
    DOCS_SUBDIR,
    SITE_BASE_URL,
)

structlogger = structlog.get_logger()


class DocPage(BaseModel):
    """A single documentation page."""

    doc_id: str = Field(
        description="Relative path-based ID, e.g. 'pro/build/writing-flows'"
    )
    url: str = Field(description="Full canonical URL on rasa.com")
    title: str = Field(description="Page title from frontmatter or filename")
    content: str = Field(description="Cleaned markdown body (no frontmatter, no JSX)")


class DocIndex(BaseModel):
    """Index of all documentation pages in the repo."""

    pages: list[DocPage]
    docs_repo_commit: str = Field(description="Git SHA of the docs repo at index time")
    created_at: str = Field(description="ISO timestamp of index creation")

    def save_to_jsonl(self, path: Path) -> None:
        """Save the index to a JSONL file (one page per line)."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            # First line: index metadata
            meta = {
                "docs_repo_commit": self.docs_repo_commit,
                "created_at": self.created_at,
                "page_count": len(self.pages),
            }
            f.write(json.dumps(meta, ensure_ascii=False) + "\n")
            for page in self.pages:
                f.write(page.model_dump_json() + "\n")
        structlogger.info(
            "docs_index.saved",
            path=str(path),
            page_count=len(self.pages),
        )

    @classmethod
    def load_from_jsonl(cls, path: Path) -> "DocIndex":
        """Load a DocIndex from a JSONL file."""
        with open(path) as f:
            lines = f.readlines()

        meta = json.loads(lines[0])
        pages = [DocPage.model_validate_json(line) for line in lines[1:]]

        return cls(
            pages=pages,
            docs_repo_commit=meta["docs_repo_commit"],
            created_at=meta["created_at"],
        )


def _get_git_commit(repo_path: Path) -> str:
    """Get the current HEAD commit SHA of the repo."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_path,
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        structlogger.warning("docs_index.git_commit_failed", repo_path=str(repo_path))
        return "unknown"


def _parse_frontmatter(raw: str) -> tuple[dict[str, str], str]:
    """Split frontmatter from markdown body.

    Returns (frontmatter_dict, body). If no frontmatter is found,
    returns an empty dict and the full content as body.
    """
    match = re.match(r"^---\s*\n(.*?)\n---\s*\n", raw, re.DOTALL)
    if not match:
        return {}, raw

    fm_block = match.group(1)
    body = raw[match.end() :]

    # Simple key: value parsing (handles single-line values only,
    # which covers title, id, sidebar_label — the fields we need)
    fm: dict[str, str] = {}
    for line in fm_block.splitlines():
        if ":" in line and not line.startswith(" ") and not line.startswith("\t"):
            key, _, value = line.partition(":")
            key = key.strip()
            value = value.strip().strip("\"'")
            if key and value:
                fm[key] = value

    return fm, body


def _clean_markdown(body: str) -> str:
    """Strip JSX imports, components, and MDX-specific syntax."""
    lines = body.splitlines()
    cleaned: list[str] = []

    for line in lines:
        # Skip import statements
        if re.match(r"^\s*import\s+", line):
            continue
        # Skip JSX self-closing tags like <Component ... />
        if re.match(r"^\s*<\w+[^>]*/>\s*$", line):
            continue
        cleaned.append(line)

    text = "\n".join(cleaned)

    # Remove inline JSX tags (opening and closing), keep inner content
    text = re.sub(r"<(\w+)[^>]*>", "", text)
    text = re.sub(r"</\w+>", "", text)

    # Collapse excessive blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def _derive_url_and_doc_id(
    relative_path: Path, frontmatter_id: Optional[str]
) -> Tuple[str, str]:
    """Derive the canonical URL and doc_id from file path and optional frontmatter id.

    Docusaurus uses the frontmatter `id` as the URL slug when present,
    replacing the filename. The parent directory path is preserved.

    Returns:
        Tuple of (canonical_url, doc_id).
    """
    parent = str(relative_path.parent)

    slug = relative_path.stem
    if frontmatter_id:
        slug = frontmatter_id

    if parent == ".":
        return f"{SITE_BASE_URL}/{slug}", slug

    # Normalize path separators
    parent = parent.replace("\\", "/")
    return f"{SITE_BASE_URL}/{parent}/{slug}", f"{parent}/{slug}"


def build_index(docs_repo_path: Path) -> DocIndex:
    """Build a DocIndex by parsing all .mdx files in the docs directory.

    Args:
        docs_repo_path: Root of the documentation repository
            (contains the docs/ subdirectory).

    Returns:
        DocIndex with all parsed pages.
    """
    docs_dir = docs_repo_path / DOCS_SUBDIR
    if not docs_dir.is_dir():
        raise FileNotFoundError(
            f"Docs directory not found: {docs_dir}. "
            f"Expected a '{DOCS_SUBDIR}/' subdirectory in {docs_repo_path}"
        )

    pages: list[DocPage] = []
    mdx_files = sorted(docs_dir.rglob("*.mdx"))
    for mdx_path in mdx_files:
        relative_path = mdx_path.relative_to(docs_dir)

        # skip snippets and archive files
        if any(part in DOCS_EXCLUDED_DIRS for part in relative_path.parts):
            continue

        raw = mdx_path.read_text(encoding="utf-8")
        frontmatter, body = _parse_frontmatter(raw)
        content = _clean_markdown(body)

        if not content:
            structlogger.warning(
                "docs_index.empty_page",
                path=str(relative_path),
            )
            continue

        fm_id = frontmatter.get("id")
        title = (
            frontmatter.get("title")
            or frontmatter.get("sidebar_label")
            or relative_path.stem
        )

        url, doc_id = _derive_url_and_doc_id(relative_path, fm_id)
        page = DocPage(
            doc_id=doc_id,
            url=url,
            title=title,
            content=content,
        )
        pages.append(page)

    commit = _get_git_commit(docs_repo_path)

    structlogger.info(
        "docs_index.built",
        page_count=len(pages),
        docs_repo_commit=commit,
    )

    return DocIndex(
        pages=pages,
        docs_repo_commit=commit,
        created_at=datetime.now(timezone.utc).isoformat(),
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build a documentation page index from a Docusaurus repo.",
    )
    parser.add_argument(
        "--docs-repo",
        required=True,
        help="Path to the documentation repository root.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output path for the JSONL index file.",
    )
    args = parser.parse_args()

    docs_repo_path = Path(args.docs_repo).resolve()
    output_path = Path(args.output).resolve()

    index = build_index(docs_repo_path)
    index.save_to_jsonl(output_path)

    return 0


if __name__ == "__main__":
    sys.exit(main())
