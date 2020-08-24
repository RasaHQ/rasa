from pathlib import Path
from typing import List, Text
import re

import pytest

from rasa.nlu.training_data.formats import RasaYAMLReader


DOCS_BASE_DIR = Path("docs/")
MDX_DOCS_FILES = list((DOCS_BASE_DIR / "docs").glob("**/*.mdx"))
# we're matching codeblocks with either `yaml-rasa` or `yml-rasa` types
# we support title or no title (you'll get a nice error message if there is a title)
TRAINING_DATA_CODEBLOCK_RE = re.compile(
    r"```y(?:a)?ml-rasa(?: title=[\"'][^\"']+[\"'])?(?: \((?P<yaml_path>.+?)\))?[^\n]*\n(?P<codeblock>.*?)```",
    re.DOTALL,
)


@pytest.mark.parametrize("mdx_file_path", MDX_DOCS_FILES)
def test_docs_training_data(mdx_file_path: Path):
    with mdx_file_path.open("r") as handle:
        mdx_content = handle.read()

    matches = TRAINING_DATA_CODEBLOCK_RE.finditer(mdx_content)
    lines_with_errors: List[Text] = []

    for match in matches:
        yaml_path = match.group("yaml_path")
        if yaml_path:
            with (DOCS_BASE_DIR / yaml_path).open("r") as handle:
                codeblock = handle.read()
        else:
            codeblock = match.group("codeblock")

        start_index = match.span()[0]
        line_number = mdx_content.count("\n", 0, start_index) + 1
        try:
            RasaYAMLReader.validate(codeblock)
        except ValueError:
            lines_with_errors.append(str(line_number))

    if lines_with_errors:
        raise AssertionError(
            f"({mdx_file_path}): Invalid training data found "
            f"at line{'s' if len(lines_with_errors) > 1 else ''} {', '.join(lines_with_errors)}"
        )
