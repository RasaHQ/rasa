import subprocess

VALIDATE_COMMAND = "rasa data validate --data data/test/test_integration \
\-d data/test/test_integration/domain.yml"


# NOTE this will be extended to test cli logs at process run to validate log level
def capture(command):
    proc = subprocess.Popen(command, stdout=subprocess.PIPE, shell=True)
    out = proc.communicate()[0].decode("utf-8")
    return out, proc.returncode


def test_rasa_validate_debug_no_errors():
    # Test captures the subprocess output for the command run
    # validates that the data in 'data/test/test_integration' throws no cli errors
    # in 'debug' mode
    command = [VALIDATE_COMMAND, "--debug"]
    out, exitcode = capture(command)
    assert exitcode == 0
    assert out == ""


def test_rasa_validate_verbose_no_errors():
    # Test captures the subprocess output for the command run
    # and validates that the data in 'data/test/test_integration' throws no cli errors
    # in 'verbose' mode
    command = [VALIDATE_COMMAND, "--verbose"]
    out, exitcode = capture(command)
    assert exitcode == 0
    assert out == ""


def test_rasa_validate_quiet_no_errors():
    # Test captures the subprocess output for the command run
    # and validates that the data in 'data/test/test_integration' throws no cli errors
    # in 'quiet' mode
    command = [VALIDATE_COMMAND, "--quiet"]
    out, exitcode = capture(command)
    assert exitcode == 0
    assert out == ""
