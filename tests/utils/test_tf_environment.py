import pytest
from rasa.utils.tensorflow.environment import parse_gpu_config


@pytest.mark.parametrize(
    "gpu_config_string, parsed_gpu_config",
    [("0: 1024", {0: 1024}), ("0:1024, 1:2048", {0: 1024, 1: 2048})],
)
def test_gpu_config_parser(gpu_config_string, parsed_gpu_config):
    assert parse_gpu_config(gpu_config_string) == parsed_gpu_config
