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
    "messages",
    [
        [],
        [
            Message({TEXT: "message 1", INTENT: {}, ENTITIES: [{}], "message_id": 1,}),
            Message({TEXT: "message 2", INTENT: {}, "message_id": 2, "metadata": {}}),
        ],
    ],
)
def test_prediction_adder_add_message(
    default_model_storage: ModelStorage,
    default_execution_context: ExecutionContext,
    config_path: Text,
    domain_path: Text,
    messages: List[Message],
):
    component = NLUPredictionToHistoryAdder.create(
        {**NLUPredictionToHistoryAdder.get_default_config()},
        default_model_storage,
        Resource("test"),
        default_execution_context,
    )

    tracker = DialogueStateTracker("test", None)
    domain = Domain.from_file(path="data/test_domains/travel_form.yml")
    original_message = UserMessage(text="hello", input_channel="slack")
    tracker = component.add(messages, tracker, domain, original_message)

    assert len(tracker.events) == len(messages)
    for i, message in enumerate(messages):
        assert isinstance(tracker.events[i], UserUttered)
        assert tracker.events[i].text == message.data.get(TEXT)
        assert tracker.events[i].intent == message.data.get(INTENT)
        assert tracker.events[i].entities == message.data.get(ENTITIES, [])
        assert tracker.events[i].input_channel == original_message.input_channel
        assert tracker.events[i].message_id == message.data.get("message_id")
        assert tracker.events[i].metadata == message.data.get("metadata", {})
