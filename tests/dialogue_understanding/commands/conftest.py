import pytest

from rasa.dialogue_understanding.commands import StartFlowCommand
from rasa.dialogue_understanding.processor.command_processor import execute_commands
from rasa.shared.core.events import UserUttered
from rasa.shared.core.flows.flow import FlowsList
from rasa.shared.core.trackers import DialogueStateTracker
from rasa.shared.nlu.constants import COMMANDS
from rasa.shared.core.flows.yaml_flows_io import flows_from_str


@pytest.fixture
def all_flows() -> FlowsList:
    return flows_from_str(
        """
        flows:
          foo:
            steps:
            - id: first_step
              action: action_listen
          bar:
            steps:
            - id: also_first_step
              action: action_listen
        """
    )


start_foo_user_uttered = UserUttered(
    "start foo", None, None, {COMMANDS: [StartFlowCommand("foo").as_dict()]}
)

start_bar_user_uttered = UserUttered(
    "start bar", None, None, {COMMANDS: [StartFlowCommand("bar").as_dict()]}
)


@pytest.fixture
def tracker(all_flows: FlowsList) -> DialogueStateTracker:
    # Creates a useful tracker that has a started flow and the current flows hashed
    tracker = DialogueStateTracker.from_events("test", evts=[start_foo_user_uttered])
    execute_commands(tracker, all_flows)
    return tracker
