import os
import sys
from unittest import mock

import pytest

sys.path.append(".github/scripts")
import validate_cpu  # noqa: E402
import validate_gpus  # noqa: E402

ENV_VARS = {
    "CUDA_VISIBLE_DEVICES": "-1",
}


@mock.patch.dict(os.environ, ENV_VARS, clear=True)
def test_validate_cpu_succeeds_when_there_are_no_gpus():
    validate_cpu.check_gpu_not_available()


@mock.patch.dict(os.environ, ENV_VARS, clear=True)
def test_validate_gpus_exits_when_there_are_no_gpus():
    # This unit test assumes that unit tests are run on a CPU
    with pytest.raises(SystemExit) as pytest_wrapped_e:
        validate_gpus.check_gpu_available()
    assert pytest_wrapped_e.type == SystemExit
    assert pytest_wrapped_e.value.code == 1
