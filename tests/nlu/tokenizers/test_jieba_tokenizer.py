from unittest.mock import patch

from rasa.nlu.constants import CLS_TOKEN


def test_jieba():
    from rasa.nlu.tokenizers.jieba_tokenizer import JiebaTokenizer

    tk = JiebaTokenizer()

    assert [t.text for t in tk.tokenize("我想去吃兰州拉面")] == [
        "我",
        "想",
        "去",
        "吃",
        "兰州",
        "拉面",
        CLS_TOKEN,
    ]

    assert [t.start for t in tk.tokenize("我想去吃兰州拉面")] == [0, 1, 2, 3, 4, 6, 9]

    assert [t.text for t in tk.tokenize("Micheal你好吗？")] == [
        "Micheal",
        "你好",
        "吗",
        "？",
        CLS_TOKEN,
    ]

    assert [t.start for t in tk.tokenize("Micheal你好吗？")] == [0, 7, 9, 10, 12]


def test_jieba_load_dictionary(tmpdir_factory):
    from rasa.nlu.tokenizers.jieba_tokenizer import JiebaTokenizer

    dictionary_path = tmpdir_factory.mktemp("jieba_custom_dictionary").strpath

    component_config = {"dictionary_path": dictionary_path}

    with patch.object(
        JiebaTokenizer, "load_custom_dictionary", return_value=None
    ) as mock_method:
        tk = JiebaTokenizer(component_config)
        tk.tokenize("")

    mock_method.assert_called_once_with(dictionary_path)
