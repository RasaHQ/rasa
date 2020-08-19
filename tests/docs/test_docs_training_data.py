from pathlib import Path
from typing import Text
import re

import pytest

from rasa.nlu.training_data.formats import RasaYAMLReader


MDX_DOCS_FILES = Path("docs/docs").glob("**/*.mdx")
# we're matching codeblocks with either `yaml-rasa` or `yml-rasa` types
# we support title or no title (you'll get a nice error message if there is a title)
TRAINING_DATA_CODEBLOCK_RE = re.compile(
    r"```y(?:a)?ml-rasa(?: title=[\"'](?P<title>[^\"']+)[\"'])?[^\n]*\n(?P<codeblock>.+?)```",
    re.DOTALL,
)


@pytest.mark.parametrize("mdx_file_path", list(MDX_DOCS_FILES))
def test_docs_training_data(mdx_file_path: Path):
    with mdx_file_path.open("r") as handle:
        mdx_content = handle.read()

    matches = TRAINING_DATA_CODEBLOCK_RE.finditer(mdx_content)

    for match in matches:
        title = match.group("title")
        codeblock = match.group("codeblock")
        try:
            RasaYAMLReader.validate(codeblock)
        except ValueError as e:
            error_message = "Invalid training data found in file"
            if title:
                error_message = f'{error_message}, in block with title "{title}"'
            raise AssertionError(error_message) from e
