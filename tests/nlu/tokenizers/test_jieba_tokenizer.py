from unittest.mock import patch

from rasa.nlu.constants import CLS_TOKEN


def test_jieba():
    from rasa.nlu.tokenizers.jieba_tokenizer import JiebaTokenizer

    component_config = {"use_cls_token": False}

    tk = JiebaTokenizer(component_config)

    assert [t.text for t in tk.tokenize("我想去吃兰州拉面")] == ["我", "想", "去", "吃", "兰州", "拉面"]

    assert [t.offset for t in tk.tokenize("我想去吃兰州拉面")] == [0, 1, 2, 3, 4, 6]

    assert [t.text for t in tk.tokenize("Micheal你好吗？")] == ["Micheal", "你好", "吗", "？"]

    assert [t.offset for t in tk.tokenize("Micheal你好吗？")] == [0, 7, 9, 10]


def test_jieba_load_dictionary(tmpdir_factory):
    from rasa.nlu.tokenizers.jieba_tokenizer import JiebaTokenizer

    dictionary_path = tmpdir_factory.mktemp("jieba_custom_dictionary").strpath

    component_config = {"dictionary_path": dictionary_path, "use_cls_token": False}

    with patch.object(
        JiebaTokenizer, "load_custom_dictionary", return_value=None
    ) as mock_method:
        tk = JiebaTokenizer(component_config)
        tk.tokenize("")

    mock_method.assert_called_once_with(dictionary_path)


def test_jieba_add_cls_token():
    from rasa.nlu.tokenizers.jieba_tokenizer import JiebaTokenizer

    component_config = {"use_cls_token": True}

    tk = JiebaTokenizer(component_config)

    assert [t.text for t in tk.tokenize("Micheal你好吗？")] == [
        "Micheal",
        "你好",
        "吗",
        "？",
        CLS_TOKEN,
    ]

    assert [t.offset for t in tk.tokenize("Micheal你好吗？")] == [0, 7, 9, 10, 12]
