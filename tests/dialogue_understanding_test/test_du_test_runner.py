from typing import Any, List, Optional
from unittest.mock import MagicMock

import pytest
from _pytest.monkeypatch import MonkeyPatch

from rasa.core.agent import Agent
from rasa.core.channels import CollectingOutputChannel, OutputChannel, UserMessage
from rasa.dialogue_understanding.commands import SetSlotCommand, StartFlowCommand
from rasa.dialogue_understanding.utils import set_record_commands_and_prompts
from rasa.dialogue_understanding_test.constants import ACTOR_USER
from rasa.dialogue_understanding_test.du_test_case import (
    DialogueUnderstandingTestCase,
    DialogueUnderstandingTestStep,
)
from rasa.dialogue_understanding_test.du_test_runner import (
    DialogueUnderstandingTestRunner,
)
from rasa.shared.core.events import BotUttered, Event, UserUttered
from rasa.shared.core.trackers import DialogueStateTracker
from rasa.shared.nlu.constants import PREDICTED_COMMANDS, PROMPTS
from tests.core.test_auth_retry_tracker_store import AsyncMock


def mock_du_test_runner_init(self: Any, *args: Any, **kwargs: Any) -> None:
    self.agent = Agent()
    processor = AsyncMock()

    # using the actual tracker store instead of a mocked one
    async def mock_fetch_tracker(
        sender_id: str,
        output_channel: Optional[OutputChannel] = None,
    ) -> Any:
        return await self.agent.tracker_store.get_or_create_tracker(sender_id)

    processor.fetch_tracker_with_initial_session = mock_fetch_tracker
    self.agent.processor = processor


@pytest.fixture
def mock_du_test_runner(monkeypatch: MonkeyPatch) -> DialogueUnderstandingTestRunner:
    monkeypatch.setattr(
        "rasa.dialogue_understanding_test.du_test_runner.DialogueUnderstandingTestRunner.__init__",
        mock_du_test_runner_init,
    )

    async def mock_handle_message(self: Any, message: Any) -> None:
        tracker = await self.tracker_store.get_or_create_tracker(message.sender_id)
        tracker.update(UserUttered(message.text))
        await self.tracker_store.save(tracker)

    monkeypatch.setattr("rasa.core.agent.Agent.handle_message", mock_handle_message)

    return DialogueUnderstandingTestRunner()


def test_get_dialogue_understanding_output_no_events(
    mock_du_test_runner: DialogueUnderstandingTestRunner,
):
    tracker = DialogueStateTracker("sender", [])
    result = mock_du_test_runner.get_dialogue_understanding_output(tracker, 0)
    assert result is None


def test_get_dialogue_understanding_output_no_user_uttered_events(
    mock_du_test_runner: DialogueUnderstandingTestRunner,
):
    tracker = DialogueStateTracker("sender", [])
    tracker.update(BotUttered(text="hi"))
    result = mock_du_test_runner.get_dialogue_understanding_output(tracker, 0)
    assert result is None


def test_get_dialogue_understanding_output_with_user_uttered_events(
    mock_du_test_runner: DialogueUnderstandingTestRunner,
):
    commands = {
        "MultiStepLLMCommandGenerator": [
            SetSlotCommand(name="slot_name", value="slot_value").as_dict(),
        ],
        "NLUCommandAdapter": [
            StartFlowCommand("test_flow").as_dict(),
        ],
    }
    prompts = {
        "MultiStepLLMCommandGenerator": [
            (
                "fill_slots_prompt",
                {
                    "user_prompt": "<prompt content>",
                    "system_prompt": "<prompt content>",
                },
            ),
            (
                "handle_flows_prompt",
                {
                    "user_prompt": "<prompt content>",
                    "system_prompt": "<prompt content>",
                },
            ),
        ],
    }

    user_uttered_event = UserUttered(
        text="hi",
        parse_data={
            "text": "hi",
            PREDICTED_COMMANDS: commands,
            PROMPTS: prompts,
        },
    )
    tracker = DialogueStateTracker("sender", [])
    tracker.update(user_uttered_event)

    result = mock_du_test_runner.get_dialogue_understanding_output(tracker, 0)

    assert result.commands == {
        "MultiStepLLMCommandGenerator": [
            SetSlotCommand(name="slot_name", value="slot_value"),
        ],
        "NLUCommandAdapter": [
            StartFlowCommand("test_flow"),
        ],
    }
    assert result.prompts == prompts


@pytest.mark.asyncio
async def test_create_tracker_for_user_step(
    mock_du_test_runner: DialogueUnderstandingTestRunner,
):
    test_runner = DialogueUnderstandingTestRunner()
    test_runner.agent = AsyncMock()

    mock_tracker = MagicMock(DialogueStateTracker)
    mock_tracker.copy.return_value = mock_tracker
    mock_tracker.events = [MagicMock(Event, timestamp=i) for i in range(5)]

    step_sender_id = "test_sender_id"
    index_user_uttered_event = 3

    await test_runner._create_tracker_for_user_step(
        step_sender_id, mock_tracker, index_user_uttered_event
    )

    assert mock_tracker.sender_id == step_sender_id
    mock_tracker.travel_back_in_time.assert_called_once_with(
        mock_tracker.events[index_user_uttered_event - 1].timestamp
    )


@pytest.mark.parametrize(
    "events, index, expected_result",
    (
        ([], 0, None),
        ([UserUttered(), BotUttered()], 5, None),
        ([UserUttered(), BotUttered()], 1, None),
        ([UserUttered("a"), BotUttered(), UserUttered("b")], 2, UserUttered("b")),
    ),
)
def test_get_user_uttered_event_from_tracker(
    events: List[Event], index: int, expected_result: Event
):
    tracker = MagicMock(DialogueStateTracker)
    tracker.events = events

    result = DialogueUnderstandingTestRunner._get_user_uttered_event_from_tracker(
        tracker, index
    )

    assert result == expected_result


@pytest.mark.asyncio
async def test_send_user_message(mock_du_test_runner: DialogueUnderstandingTestRunner):
    mock_agent = MagicMock()
    mock_agent.handle_message = AsyncMock()
    mock_du_test_runner.agent = mock_agent

    sender_id = "test_sender_id"
    output_channel = CollectingOutputChannel()

    test_step = DialogueUnderstandingTestStep(actor=ACTOR_USER, text="test text")
    test_case = DialogueUnderstandingTestCase(name="test_case", steps=[test_step])

    with set_record_commands_and_prompts():
        await mock_du_test_runner._send_user_message(
            sender_id, test_case, test_step, [], output_channel
        )

    mock_agent.handle_message.assert_awaited_once()
    assert isinstance(mock_agent.handle_message.call_args[0][0], UserMessage)
    assert mock_agent.handle_message.call_args[0][0].sender_id == sender_id
    assert mock_agent.handle_message.call_args[0][0].text == test_step.text
    assert mock_agent.handle_message.call_args[0][0].output_channel == output_channel
    assert mock_agent.handle_message.call_args[0][0].metadata == {}
