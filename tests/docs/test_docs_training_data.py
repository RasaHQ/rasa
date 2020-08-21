from pathlib import Path
from typing import Text
import re

import pytest

from rasa.nlu.training_data.formats import RasaYAMLReader


MDX_DOCS_FILES = Path("docs/docs").glob("**/*.mdx")
# we're matching codeblocks with either `yaml-rasa` or `yml-rasa` types
# we support title or no title (you'll get a nice error message if there is a title)
TRAINING_DATA_CODEBLOCK_RE = re.compile(
    r"```y(?:a)?ml-rasa(?: title=[\"'][^\"']+[\"'])?[^\n]*\n(?P<codeblock>.+?)```",
    re.DOTALL,
)


@pytest.mark.parametrize("mdx_file_path", list(MDX_DOCS_FILES))
def test_docs_training_data(mdx_file_path: Path):
    with mdx_file_path.open("r") as handle:
        mdx_content = handle.read()

    matches = TRAINING_DATA_CODEBLOCK_RE.finditer(mdx_content)

    for match in matches:
        codeblock = match.group("codeblock")
        start_index = match.span()[0]
        line_number = mdx_content.count("\n", 0, start_index) + 1
        try:
            RasaYAMLReader.validate(codeblock)
        except ValueError as e:
            raise AssertionError(
                f"({mdx_file_path}): Invalid training data found at line {line_number}"
            ) from e
