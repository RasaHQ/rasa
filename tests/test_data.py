import glob
import os

from pathlib import Path

from rasa.data import is_story_file
from rasa.utils.io import write_text_file


def test_story_file_can_not_be_yml(tmpdir: Path):
    p = tmpdir / "test_non_md.yml"
    Path(p).touch()
    assert is_story_file(str()) is False


def test_empty_story_file_is_not_story_file(tmpdir: Path):
    p = tmpdir / "test_non_md.md"
    Path(p).touch()
    assert is_story_file(str(p)) is False


def test_story_file_with_minimal_story_is_story_file(tmpdir: Path):
    p = tmpdir / "story.md"
    s = """
## my story
    """
    write_text_file(s, p)
    assert is_story_file(str(p))


def test_default_story_files_are_story_files():
    for fn in glob.glob(os.path.join("data", "test_stories", "*")):
        assert is_story_file(fn)
