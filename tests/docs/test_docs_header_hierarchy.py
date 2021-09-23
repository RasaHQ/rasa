from pathlib import Path
from typing import List, Text, Tuple

import pytest


DOCS_BASE_DIR = Path("docs/")
MDX_DOCS_FILES = list((DOCS_BASE_DIR / "docs").glob("**/*.mdx"))
CODE_BLOCK_OPEN = ["```", "<pre><code"]
CODE_BLOCK_CLOSE = ["```", "</code></pre>"]


def get_heading_level(heading: Text) -> int:
    return len(heading.split(" ", 1)[0])


def get_heading(heading: Text) -> Text:
    return heading.split(" ", 1)[1]


def line_opens_code_block(line: Text) -> bool:
    for open_tag in CODE_BLOCK_OPEN:
        if line.startswith(open_tag):
            return True
    return False


def line_closes_code_block(line: Text) -> bool:
    for close_tag in CODE_BLOCK_CLOSE:
        if line.endswith(close_tag):
            return True
    return False


@pytest.mark.parametrize("mdx_file_path", MDX_DOCS_FILES)
def test_docs_header_hierarchy(mdx_file_path: Path):
    with mdx_file_path.open("r") as handle:
        mdx_content = handle.read()

    headings: List[Tuple[int, Text, Text]] = []
    inside_code_block = False
    for idx, line in enumerate(mdx_content.split("\n")):
        line = line.strip()
        if not inside_code_block and line_opens_code_block(line):
            inside_code_block = True
        elif inside_code_block and line_closes_code_block(line):
            inside_code_block = False

        if not inside_code_block and line.startswith("#"):
            headings.append((idx + 1, get_heading(line), get_heading_level(line)))

    errors: List[Text] = []
    prev_level = 1
    for line_number, title, level in headings:
        if level == 1:
            errors.append(
                f"\n  - Title '# {title}' at line {line_number} "
                f"cannot be level 1. Use '## {title}' instead."
            )

        if level > prev_level + 1:
            errors.append(
                f"\n  - Title '{'#' * level} {title}' at line {line_number} "
                f"is skipping a level. Use '{'#' * (prev_level + 1)} {title}' instead."
            )
            # that way we continue to have errors
            prev_level = prev_level + 1
        else:
            prev_level = level

    if errors:
        raise AssertionError(f"({mdx_file_path}):{''.join(errors)}")
