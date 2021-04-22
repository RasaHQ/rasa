import pytest
from rasa.shared.nlu.training_data.formats import NLGMarkdownReader, NLGMarkdownWriter


def test_markdow_nlg_read_newlines():
    md = """
## Ask something
* faq/ask_something
  - Super answer in 2\\nlines
    """
    reader = NLGMarkdownReader()
    result = reader.reads(md)

    assert result.responses == {
        "faq/ask_something": [{"text": "Super answer in 2\nlines"}]
    }


def test_markdown_reading_deprecation():
    with pytest.warns(FutureWarning):
        NLGMarkdownReader()


def test_skip_markdown_reading_deprecation():
    with pytest.warns(None) as warnings:
        NLGMarkdownReader(ignore_deprecation_warning=True)

    assert not warnings


def test_markdown_writing_deprecation():
    with pytest.warns(FutureWarning):
        NLGMarkdownWriter()


def test_skip_markdown_writing_deprecation():
    with pytest.warns(None) as warnings:
        NLGMarkdownWriter(ignore_deprecation_warning=True)

    assert not warnings
