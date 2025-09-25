import pytest

# Skip all tests in this file if TensorFlow is not available
from rasa.utils.tensorflow import TENSORFLOW_AVAILABLE

if not TENSORFLOW_AVAILABLE:
    pytest.skip("TensorFlow is not available", allow_module_level=True)

from rasa.utils.tensorflow.exceptions import TFLayerConfigException
from rasa.utils.tensorflow.transformer import MultiHeadAttention


def test_valid_transformer_size():
    mha = MultiHeadAttention(units=256, num_heads=4)
    assert mha.units == 256
    with pytest.raises(TFLayerConfigException):
        MultiHeadAttention(units=50, num_heads=4)
