import pytest
from _pytest.monkeypatch import MonkeyPatch
from typing import Text, Dict
import multiprocessing
from rasa.utils.tensorflow.environment import setup_cpu_environment
from rasa.utils.tensorflow.environment import parse_gpu_config
from rasa.constants import ENV_CPU_INTER_OP_CONFIG, ENV_CPU_INTRA_OP_CONFIG


def tf_cpu_setter(
    inter_op_config: Text, intra_op_config: Text, shared_context_output: Dict[Text, int]
):

    monkeypatch = MonkeyPatch()
    monkeypatch.setenv(ENV_CPU_INTRA_OP_CONFIG, intra_op_config)
    monkeypatch.setenv(ENV_CPU_INTER_OP_CONFIG, inter_op_config)

    set_inter_op_val, set_intra_op_val = setup_cpu_environment()

    shared_context_output[ENV_CPU_INTER_OP_CONFIG] = set_inter_op_val
    shared_context_output[ENV_CPU_INTRA_OP_CONFIG] = set_intra_op_val


def test_tf_cpu_setting():

    shared_context_output = multiprocessing.Manager().dict()

    child_process = multiprocessing.Process(
        target=tf_cpu_setter, args=("3", "2", shared_context_output)
    )
    child_process.start()
    child_process.join()

    assert shared_context_output[ENV_CPU_INTER_OP_CONFIG] == 3
    assert shared_context_output[ENV_CPU_INTRA_OP_CONFIG] == 2


@pytest.mark.parametrize(
    "gpu_config_string, parsed_gpu_config",
    [("0: 1024", {0: 1024}), ("0:1024, 1:2048", {0: 1024, 1: 2048})],
)
def test_gpu_config_parser(gpu_config_string, parsed_gpu_config):

    assert parse_gpu_config(gpu_config_string) == parsed_gpu_config
