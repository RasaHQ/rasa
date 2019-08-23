import subprocess
import sys
import pytest
from typing import Text

NUMBER_OF_MEASUREMENTS = 15

# Maximum expected import time for rasa module when running on a Travis VM.
# Keep in mind the hardware configuration where tests are run:
# https://docs.travis-ci.com/user/reference/overview/
MAX_IMPORT_TIME_S = 0.3


def _average_import_time(n: int, module: Text) -> float:
    total = 0

    for _ in range(n):
        lines = subprocess.getoutput(
            '{} -X importtime -c "import {}"'.format(sys.executable, module)
        ).splitlines()

        parts = lines[-1].split("|")
        if parts[-1].strip() != module:
            raise Exception("Import time not found for {}.".format(module))

        total += int(parts[1].strip()) / 1000000

    return total / n


@pytest.mark.skipif(sys.version_info < (3, 7), reason="Need 3.7+ for -X importtime")
def test_import_time():
    import_time = _average_import_time(NUMBER_OF_MEASUREMENTS, "rasa")
    assert import_time < MAX_IMPORT_TIME_S
