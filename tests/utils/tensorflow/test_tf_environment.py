import pytest
from typing import Text, Dict
from rasa.utils.tensorflow.environment import _parse_gpu_config


@pytest.mark.parametrize(
    "gpu_config_string, parsed_gpu_config",
    [("0: 1024", {0: 1024}), ("0:1024, 1:2048", {0: 1024, 1: 2048})],
)
def test_gpu_config_parser(gpu_config_string: Text, parsed_gpu_config: Dict[int, int]):
    assert _parse_gpu_config(gpu_config_string) == parsed_gpu_config
