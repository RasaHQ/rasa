import pytest
import tensorflow as tf
from rasa.utils.tensorflow.environment import setup_cpu_environment
from rasa.utils.tensorflow.environment import parse_gpu_config
from _pytest.monkeypatch import MonkeyPatch
from rasa.constants import ENV_CPU_INTER_OP_CONFIG, ENV_CPU_INTRA_OP_CONFIG


def test_tf_cpu_environment_setting():

    monkeypatch = MonkeyPatch()
    monkeypatch.setenv(ENV_CPU_INTRA_OP_CONFIG, "3")
    monkeypatch.setenv(ENV_CPU_INTER_OP_CONFIG, "2")

    setup_cpu_environment()

    assert tf.config.threading.get_inter_op_parallelism_threads() == 2
    assert tf.config.threading.get_intra_op_parallelism_threads() == 3


@pytest.mark.parametrize(
    "gpu_config_string, parsed_gpu_config",
    [("0: 1024", {0: 1024}), ("0:1024, 1:2048", {0: 1024, 1: 2048})],
)
def test_parsed_gpu_config(gpu_config_string, parsed_gpu_config):

    assert parse_gpu_config(gpu_config_string) == parsed_gpu_config
