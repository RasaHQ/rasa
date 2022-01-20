import contextlib
import subprocess
import shutil
import uuid

from pathlib import Path
from typing import Text
from rasa.core.processor import MessageProcessor
from rasa.core.channels.channel import CollectingOutputChannel, UserMessage

server_states = ["Action endpoint is up and running on", "Failed to register package"]


@contextlib.contextmanager
def run_sdk_for_rasa_project(project_dir: Text):
    args = [shutil.which("rasa"), "run", "actions"]
    process = subprocess.Popen(args, stderr=subprocess.PIPE, cwd=project_dir)
    try:
        yield process
    finally:
        process.terminate()


def start_sdk_server(process):
    read_lines = []
    while True:
        line = read_process_line(process)
        read_lines.append(line)
        if any(state in line for state in server_states):
            break
    return read_lines


def check_condition(sdk_process_lines, condition):
    return any(condition in line for line in sdk_process_lines)


def read_process_line(process) -> Text:
    return process.stderr.readline().decode("utf-8")


def test_action_server_start_in_empty_dir():
    temp_dir = Path.cwd()
    with run_sdk_for_rasa_project(str(temp_dir)) as process:
        read_lines = start_sdk_server(process)
        assert check_condition(read_lines, "Failed to register package")


async def test_action_server_use_formbot(
    formbot_project: Text, default_processor: MessageProcessor
):
    with run_sdk_for_rasa_project(formbot_project) as process:
        read_lines = start_sdk_server(process)
        assert check_condition(read_lines, "Action endpoint is up and running")
        assert check_condition(read_lines, "Registered function for 'validate_restaurant_form'")

        sender_id = uuid.uuid4().hex

        message = UserMessage(
            text="hi", output_channel=CollectingOutputChannel(), sender_id=sender_id,
        )
        await default_processor.handle_message(message)
        message = UserMessage(
            text="im looking for a restaurant",
            output_channel=CollectingOutputChannel(),
            sender_id=sender_id,
        )
        await default_processor.handle_message(message)

        tracker = default_processor.tracker_store.retrieve(sender_id)
        events = list(tracker.events)

        print(events)
