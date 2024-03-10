import pytest

from rasa.utils.tensorflow.transformer import MultiHeadAttention


def test_valid_transformer_size():
    mha = MultiHeadAttention(units=256, num_heads=4)
    assert mha.units == 256
    with pytest.raises(SystemExit):
        MultiHeadAttention(units=50, num_heads=4)
