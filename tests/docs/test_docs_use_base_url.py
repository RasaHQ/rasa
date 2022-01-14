from pathlib import Path
from typing import List, Text
import re

import pytest


DOCS_BASE_DIR = Path("docs/")
MDX_DOCS_FILES = list((DOCS_BASE_DIR / "docs").glob("**/*.mdx"))
# we're matching anchors with href containing strings, but not starting
# with "http". This also exclude local href already configured using `useBaseUrl()`
ANCHOR_RE = re.compile(
    r"<(a|Button)[^>]*href=\"(?P<href>(?!http).+?)\"[^>]*>", re.DOTALL
)


@pytest.mark.parametrize("mdx_file_path", MDX_DOCS_FILES)
def test_docs_anchors_base_url(mdx_file_path: Path):
    with mdx_file_path.open("r") as handle:
        mdx_content = handle.read()

    matches = ANCHOR_RE.finditer(mdx_content)
    lines_with_errors: List[Text] = []

    for match in matches:
        href = match.group("href")
        start_index = match.span()[0]
        line_number = mdx_content.count("\n", 0, start_index) + 1
        lines_with_errors.append(f"{line_number} with href={href}")

    if lines_with_errors:
        plural = "s" if len(lines_with_errors) > 1 else ""
        raise AssertionError(
            f"({mdx_file_path}): Invalid anchor{plural} not using useBaseUrl() "
            f"at line{plural} {', '.join(lines_with_errors)}"
        )
