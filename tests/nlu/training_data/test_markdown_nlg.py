from rasa.nlu.training_data.formats import NLGMarkdownReader


def test_markdownnlg_read_newlines():
    md = """
## Ask something
* faq/ask_something
  - Super answer in 2\\nlines
    """
    reader = NLGMarkdownReader()
    result = reader.reads(md)

    assert result.nlg_stories == {"faq/ask_something": ["Super answer in 2\nlines"]}
