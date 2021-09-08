from rasa.shared.core.events import UserUttered
from rasa.shared.nlu.constants import ENTITIES, INTENT, TEXT
from rasa.core.channels.channel import UserMessage
import pytest
from typing import Text, List
from rasa.shared.core.trackers import DialogueStateTracker
from rasa.shared.nlu.training_data.message import Message
from rasa.engine.graph import ExecutionContext
from rasa.engine.storage.resource import Resource
from rasa.engine.storage.storage import ModelStorage
from rasa.shared.core.domain import Domain
from rasa.graph_components.adders.nlu_prediction_to_history_adder import (
    NLUPredictionToHistoryAdder,
)


@pytest.mark.parametrize(
    "messages,expected,input_channel",
    [
        ([], [], "slack"),
        (
            [
                Message(
                    {TEXT: "message 1", INTENT: {}, ENTITIES: [{}], "message_id": "1",}
                ),
                Message(
                    {TEXT: "message 2", INTENT: {}, "message_id": "2", "metadata": {}}
                ),
            ],
            [
                UserUttered("message 1", {}, [{}], None, None, "slack", "1"),
                UserUttered("message 2", {}, None, None, None, "slack", "2", {}),
            ],
            "slack",
        ),
    ],
)
def test_prediction_adder_add_message(
    default_model_storage: ModelStorage,
    default_execution_context: ExecutionContext,
    messages: List[Message],
    expected: List[UserUttered],
    input_channel: Text,
):
    component = NLUPredictionToHistoryAdder.create(
        {**NLUPredictionToHistoryAdder.get_default_config()},
        default_model_storage,
        Resource("test"),
        default_execution_context,
    )

    tracker = DialogueStateTracker("test", None)
    domain = Domain.from_file(path="data/test_domains/travel_form.yml")
    original_message = UserMessage(text="hello", input_channel=input_channel)
    tracker = component.add(messages, tracker, domain, original_message)

    assert len(tracker.events) == len(messages)
    for i, message in enumerate(messages):
        assert isinstance(tracker.events[i], UserUttered)
        assert tracker.events[i].text == expected[i].text
        assert tracker.events[i].intent == expected[i].intent
        assert tracker.events[i].entities == expected[i].entities
        assert tracker.events[i].input_channel == expected[i].input_channel
        assert tracker.events[i].message_id == expected[i].message_id
        assert tracker.events[i].metadata == expected[i].metadata
