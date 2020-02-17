import pytest
from _pytest.monkeypatch import MonkeyPatch
from typing import Text, Dict
import multiprocessing
from rasa.utils.tensorflow.environment import setup_cpu_environment
from rasa.utils.tensorflow.environment import parse_gpu_config
from rasa.constants import ENV_CPU_INTER_OP_CONFIG, ENV_CPU_INTRA_OP_CONFIG


@pytest.mark.parametrize(
    "gpu_config_string, parsed_gpu_config",
    [("0: 1024", {0: 1024}), ("0:1024, 1:2048", {0: 1024, 1: 2048})],
)
def test_gpu_config_parser(gpu_config_string, parsed_gpu_config):

    assert parse_gpu_config(gpu_config_string) == parsed_gpu_config
