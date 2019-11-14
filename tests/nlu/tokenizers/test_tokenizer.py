import pytest


def test_tokens_comparison():
    from rasa.nlu.tokenizers.tokenizer import Token

    x = Token("hello", 0)
    y = Token("Hello", 0)

    assert x == x
    assert y < x

    assert x != 1

    with pytest.raises(TypeError):
        assert y < "a"
