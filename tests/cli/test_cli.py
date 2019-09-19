import pytest


@pytest.mark.repeat(3)
def test_cli_start(run):
    import time

    start = time.time()
    output = run("--help")
    end = time.time()

    duration = end - start

    assert duration < 3  # startup of cli should not take longer than 3 seconds
    assert output.ret == 0


def test_data_convert_help(run):
    output = run("--help")

    help_text = """usage: rasa [-h] [--version]
            {init,run,shell,train,interactive,test,visualize,data,x} ..."""

    lines = help_text.split("\n")

    for i, line in enumerate(lines):
        assert output.outlines[i] == line
    assert output.ret == 0


def test_return(run):
    output = run("unknown_command")
    assert output.ret != 0


def test_return_error(run):
    output = run("data", "validate", "--data non_existent_path")
    assert output.ret != 0


def test_return_error_when_error_in_logs(run_in_default_project):
    with open("domain.yml", "r") as f:
        lines = f.readlines()
    with open("domain.yml", "w") as f:
        for line in lines:
            if "affirm" not in line:
                f.write(line)

    output = run_in_default_project("data", "validate")
    assert output.ret == 1


def test_return_passed_no_error_in_logs(run_in_default_project):
    output = run_in_default_project("data", "validate")
    assert output.ret == 0
