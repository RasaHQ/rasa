import contextlib
import subprocess
import shutil
import uuid

from pathlib import Path
from typing import Text
from rasa.core.processor import MessageProcessor
from rasa.core.channels.channel import CollectingOutputChannel, UserMessage


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
        assert (
            "Registered function for 'validate_restaurant_form'"
            in read_process_line(process)
        )


async def test_action_server_use_formbot(
    formbot_project: Text, default_processor: MessageProcessor
):
    # test_update_tracker_session_with_metadata
    with run_in_rasa_project(formbot_project, ["run", "actions"]) as process:
        for i in range(3):
            # we need to make sure the server has started and
            # all the functions are registered
            read_process_line(process)

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

        # we can check the events and make sure they contain
        # the required ones

        print(events)
