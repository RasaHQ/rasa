from typing import Optional, Text, Dict
from rasa.shared.core.domain import Domain
from rasa.shared.core.training_data.structures import STEP_COUNT, StoryGraph, StoryStep
from rasa.shared.core.events import UserUttered, ActionExecuted
from rasa.architecture_prototype.graph_components import StoryToTraingingDataConverter
from rasa.shared.core.trackers import DialogueStateTracker

from rasa.shared.nlu.constants import (
    ENTITY_ATTRIBUTE_VALUE,
    ENTITY_ATTRIBUTE_TYPE,
    ENTITY_ATTRIBUTE_ROLE,
    ENTITY_ATTRIBUTE_GROUP,
    INTENT,
    TEXT,
)


def _create_entity(
    value: Text, type: Text, role: Optional[Text] = None, group: Optional[Text] = None
) -> Dict[Text, Text]:
    entity = {}
    entity[ENTITY_ATTRIBUTE_VALUE] = value
    entity[ENTITY_ATTRIBUTE_TYPE] = type
    entity[ENTITY_ATTRIBUTE_ROLE] = role
    entity[ENTITY_ATTRIBUTE_GROUP] = group
    return entity


def test_convert_for_training():
    """Tests whether story graph and domain information ends up in training data.

    Note: We do not test for completeness/shape of the result here. These aspects are
    defined by the same principles as the lookup table. Hence, these aspects should
    be tested in the lookup table tests, whereas here we just probe whether the lookup
    table is applied correctly.
    """
    # create domain and story graph
    intent_in_domain_only = "only-appears-in-domain"
    event_text_prefix = "only-appears-in-events"
    domain = Domain(
        intents=["greet", "inform", intent_in_domain_only],
        entities=["entity_name"],
        slots=[],
        responses=dict(),
        action_names=["action_listen", "utter_greet"],
        forms=dict(),
        action_texts=["Hi how are you?"],
    )
    events = [
        ActionExecuted(action_name="action_listen"),
        UserUttered(
            text=f"{event_text_prefix}-1",
            intent={"intent_name": "greet"},
            entities=[_create_entity(value="Bot", type="entity_name")],
        ),
        ActionExecuted(action_name="utter_greet", action_text="Hi how are you?"),
        ActionExecuted(action_name="action_listen"),
        UserUttered(text=f"{event_text_prefix}-2", intent={"intent_name": "inform"}),
        ActionExecuted(action_name="action_listen"),
    ]
    story_graph = StoryGraph([StoryStep(events=events)])

    # convert!
    training_data = StoryToTraingingDataConverter.convert_for_training(
        domain=domain, story_graph=story_graph
    )

    # check information from events is contained - by checking the texts
    training_texts = sorted(
        message.data.get(TEXT)
        for message in training_data.training_examples
        if TEXT in message.data
    )
    assert all(
        [
            (
                text.startswith(event_text_prefix)
                or text == StoryToTraingingDataConverter.WORKAROUND_TEXT
            )
            for text in training_texts
        ]
    )
    assert StoryToTraingingDataConverter.WORKAROUND_TEXT in training_texts
    # check information from domain is contained - by checking the intents
    training_intents = set(
        message.get(INTENT) for message in training_data.training_examples
    )
    assert intent_in_domain_only in training_intents


def test_convert_for_prediction():
    """Tests whether information from tracker events end up in the messages.

    Note: We do not test for completeness/shape of the result here. These aspects are
    defined by the same principles as the lookup table. Hence, these aspects should
    be tested in the lookup table tests, whereas here we just probe whether the lookup
    table is applied correctly.
    """
    # create tracker
    event_text_prefix = "this-is-a-text-prefix"
    events = [
        UserUttered(
            text=f"{event_text_prefix}-1",
            intent={"intent_name": "greet"},
            entities=[_create_entity(value="Bot", type="entity_name")],
        ),
        ActionExecuted(action_name="utter_greet", action_text="Hi how are you?"),
        ActionExecuted(action_name="action_listen"),
        UserUttered(text=f"{event_text_prefix}-2", intent={"intent_name": "inform"}),
    ]
    tracker = DialogueStateTracker.from_events(sender_id="arbitrary", evts=events)

    # convert!
    messages = StoryToTraingingDataConverter.convert_for_inference(tracker)

    # check information from events is contained
    message_texts = sorted(
        message.data.get(TEXT) for message in messages if TEXT in message.data
    )
    assert all([(text.startswith(event_text_prefix)) for text in message_texts])
    assert StoryToTraingingDataConverter.WORKAROUND_TEXT not in message_texts
