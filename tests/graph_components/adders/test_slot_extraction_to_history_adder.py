from rasa.core.nlg import TemplatedNaturalLanguageGenerator
from rasa.engine.graph import ExecutionContext
from rasa.engine.storage.resource import Resource
from rasa.engine.storage.storage import ModelStorage
from rasa.graph_components.adders.slot_extraction_to_history_adder import (
    SlotExtractionToHistoryAdder,
)
from rasa.utils.endpoints import EndpointConfig

from rasa.shared.core.domain import Domain
from rasa.shared.core.events import SlotSet
from rasa.shared.core.trackers import DialogueStateTracker


def test_prediction_adder_add_message(
    default_model_storage: ModelStorage, default_execution_context: ExecutionContext,
):
    component = SlotExtractionToHistoryAdder.create(
        {**SlotExtractionToHistoryAdder.get_default_config()},
        default_model_storage,
        Resource("test"),
        default_execution_context,
    )

    domain = Domain.from_dict(
        {
            "slots": {
                "some_slot": {
                    "type": "any",
                    "mappings": [{"type": "from_intent", "value": "test"}],
                }
            }
        }
    )

    tracker = DialogueStateTracker("test", None)
    nlg = TemplatedNaturalLanguageGenerator(domain.responses)
    tracker = component.add(
        tracker, domain, nlg, EndpointConfig("https://example.com/webhooks/actions")
    )

    assert SlotSet("some_slot", "test") in tracker.events
