import contextlib
import subprocess
import shutil

from pathlib import Path
from typing import Text


@contextlib.contextmanager
def run_in_rasa(project_dir: Text, command_args):
    args = [shutil.which("rasa")] + list(command_args)
    process = subprocess.Popen(args, stderr=subprocess.PIPE, cwd=project_dir)
    try:
        output = process.communicate()[1]
        if output:
            yield output.decode("utf-8")
        else:
            yield ""
    finally:
        process.terminate()


def test_action_server_start_in_empty_dir():
    temp_dir = Path.cwd()
    with run_in_rasa(str(temp_dir), ["run", "actions"]) as output:
        assert "Starting action endpoint server..." in output
        assert "Failed to register package 'actions'" in output


def test_action_server_start(simple_project: Text):
    print(simple_project)

    with run_in_rasa(simple_project, ["run", "actions"]) as output:
        print(output)
        assert "Starting action endpoint server..." in output
        assert "Action endpoint is up and running on" in output
