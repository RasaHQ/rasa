import contextlib
import os
import subprocess
import shutil

from pathlib import Path
from typing import Text


@contextlib.contextmanager
def run_in_rasa_project(project_dir: Text, command_args):
    args = [shutil.which("rasa")] + list(command_args)
    process = subprocess.Popen(args, stderr=subprocess.PIPE, cwd=project_dir)
    try:
        yield process
    finally:
        process.terminate()


def read_process_line(process) -> Text:
    return process.stderr.readline().decode("utf-8")


def test_action_server_start_in_empty_dir():
    temp_dir = Path.cwd()
    with run_in_rasa_project(str(temp_dir), ["run", "actions"]) as process:
        assert "Starting action endpoint server..." in read_process_line(process)
        assert "Failed to register package 'actions'" in read_process_line(process)


def test_action_server_start(simple_project: Text):
    with run_in_rasa_project(simple_project, ["run", "actions"]) as process:
        assert "Starting action endpoint server..." in read_process_line(process)
        assert "Action endpoint is up and running on " in read_process_line(process)


def test_action_server_start_formbot(formbot_project: Text):
    with run_in_rasa_project(formbot_project, ["run", "actions"]) as process:
        assert "Starting action endpoint server..." in read_process_line(process)
        assert "Registered function for 'validate_restaurant_form'" in read_process_line(process)