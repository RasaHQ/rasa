import subprocess
import sys
import pytest

AVG_ITERATIONS = 15

# Maximum expected import time for rasa module when running on a Travis VM.
# Keep in mind the hardware configuration where tests are run:
# https://docs.travis-ci.com/user/reference/overview/
MAX_IMPORT_TIME_S = 0.3


def average_import_time(n, module):
    total = 0

    py_cmd_version = tuple(
        int(part)
        for part in subprocess.getoutput(
            "python -c 'import sys; print(sys.version_info[:3])'"
        )
        .strip("()")
        .split(",")
    )

    if py_cmd_version < (3, 7):
        raise Exception(
            "Can't use Python version {} for profiling (required: 3.7+).".format(
                py_cmd_version
            )
        )

    for _ in range(n):
        lines = subprocess.getoutput(
            'python -X importtime -c "import {}"'.format(module)
        ).splitlines()

        parts = lines[-1].split("|")
        if parts[-1].strip() != module:
            raise Exception("Import time not found for {}.".format(module))

        total += int(parts[1].strip()) / 1000000

    return total / n


@pytest.mark.skipif(sys.version_info < (3, 7), reason="Need 3.7+ for -X importtime")
def test_import_time():
    import_time = average_import_time(AVG_ITERATIONS, "rasa")
    assert import_time < MAX_IMPORT_TIME_S
