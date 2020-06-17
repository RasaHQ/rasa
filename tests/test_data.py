import glob
import os

from pathlib import Path

from rasa.constants import DEFAULT_E2E_TESTS_PATH
from rasa import data
from rasa.utils.io import write_text_file


def test_story_file_can_not_be_yml(tmpdir: Path):
    p = tmpdir / "test_non_md.yml"
    Path(p).touch()
    assert data.is_story_file(str()) is False


def test_empty_story_file_is_not_story_file(tmpdir: Path):
    p = tmpdir / "test_non_md.md"
    Path(p).touch()
    assert data.is_story_file(str(p)) is False


def test_story_file_with_minimal_story_is_story_file(tmpdir: Path):
    p = tmpdir / "story.md"
    s = """
## my story
    """
    write_text_file(s, p)
    assert data.is_story_file(str(p))


def test_default_story_files_are_story_files():
    for fn in glob.glob(os.path.join("data", "test_stories", "*")):
        assert data.is_story_file(fn)


def test_default_conversation_tests_are_conversation_tests(tmpdir: Path):
    parent = tmpdir / DEFAULT_E2E_TESTS_PATH
    Path(parent).mkdir(parents=True)

    e2e_path = parent / "conversation_tests.md"
    e2e_story = """## my story test"""
    write_text_file(e2e_story, e2e_path)

    assert data.is_end_to_end_conversation_test_file(str(e2e_path))


def test_nlu_data_files_are_not_conversation_tests(tmpdir: Path):
    parent = tmpdir / DEFAULT_E2E_TESTS_PATH
    Path(parent).mkdir(parents=True)

    nlu_path = parent / "nlu.md"
    nlu_data = """
## intent: greet
- hello
- hi
- hallo
    """
    write_text_file(nlu_data, nlu_path)

    assert not data.is_end_to_end_conversation_test_file(str(nlu_path))


def test_domain_files_are_not_conversation_tests(tmpdir: Path):
    parent = tmpdir / DEFAULT_E2E_TESTS_PATH
    Path(parent).mkdir(parents=True)

    domain_path = parent / "domain.yml"

    assert not data.is_end_to_end_conversation_test_file(str(domain_path))
