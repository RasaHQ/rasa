from rasa.shared.nlu.training_data.features import Features
from typing import Dict, Optional, Text, List, Any
import json
import numpy as np

from rasa.shared.nlu.training_data.message import Message
from rasa.shared.core.domain import Domain
from rasa.engine.storage.resource import Resource
from rasa.engine.graph import ExecutionContext
from rasa.engine.storage.storage import ModelStorage
from rasa.nlu.classifiers.regex_message_handler import RegexMessageHandlerGraphComponent
import pytest
from rasa.shared.constants import INTENT_MESSAGE_PREFIX
from rasa.shared.nlu.constants import (
    ENTITIES,
    ENTITY_ATTRIBUTE_END,
    ENTITY_ATTRIBUTE_START,
    ENTITY_ATTRIBUTE_TYPE,
    ENTITY_ATTRIBUTE_VALUE,
    FEATURE_TYPE_SENTENCE,
    INTENT,
    INTENT_NAME_KEY,
    INTENT_RANKING_KEY,
    PREDICTED_CONFIDENCE_KEY,
    TEXT,
)


@pytest.fixture
def regex_message_handler(
    default_model_storage: ModelStorage, default_execution_context: ExecutionContext
) -> RegexMessageHandlerGraphComponent:
    return RegexMessageHandlerGraphComponent.create(
        config={},
        model_storage=default_model_storage,
        resource=Resource("unused"),
        execution_context=default_execution_context,
    )


@pytest.mark.parametrize(
    "confidence,entities,expected_confidence,expected_entities,should_warn",
    [
        # easy examples - where entities or intents might be missing
        (None, None, 1.0, [], False),
        ("0.2134345", None, 0.2134345, [], False),
        ("0", None, 0, [], False),
        (
            None,
            json.dumps({"entity1": "entity_value1", "entity2": 2.0}),
            1.0,
            [
                {
                    ENTITY_ATTRIBUTE_TYPE: "entity1",
                    ENTITY_ATTRIBUTE_VALUE: "entity_value1",
                },
                {ENTITY_ATTRIBUTE_TYPE: "entity2", ENTITY_ATTRIBUTE_VALUE: 2.0},
            ],
            False,
        ),
        # malformed confidences
        (
            "-2",
            None,
            1.0,
            [],
            True,
        ),  # no confidence string; some unidentified part left
        ("abc0.2134345", None, 1.0, [], True),  # same
        ("123", None, 1.0, [], True),  # value extracted by > 1
        ("123?", None, 1.0, [], True),  # value extracted by > 1
        ("1.0.", None, 0.0, [], True),  # confidence string extracted but not a float
        # malformed entities
        (None, json.dumps({"entity1": "entity2"}), 1.0, [], True),
        (None, '{"entity1","entity2":2.0}', 1.0, [], True),
        # ... note: if the confidence is None, the following will raise an error!
        (
            "1.0",
            json.dumps(["entity1"]),
            1.0,
            [],
            True,
        ),  # no entity string extracted; some unexpected string left
    ],
)
def test_process_unpacks_attributes_from_single_message_and_fallsback_if_needed(
    regex_message_handler: RegexMessageHandlerGraphComponent,
    confidence: Optional[Text],
    entities: Optional[Text],
    expected_confidence: float,
    expected_entities: Optional[List[Dict[Text, Any]]],
    should_warn: bool,
):

    # dummy intent
    expected_intent = "my-intent"

    # construct text according to pattern
    text = " \t  " + INTENT_MESSAGE_PREFIX + expected_intent
    if confidence is not None:
        text += f"@{confidence}"
    if entities is not None:
        text += entities
    text += " \t "

    # create a message with some dummy attributes and features
    message = Message(
        data={TEXT: text, INTENT: "extracted-from-the-pattern-text-via-nlu"},
        features=[
            Features(
                features=np.zeros((1, 1)),
                feature_type=FEATURE_TYPE_SENTENCE,
                attribute=TEXT,
                origin="nlu-pipeline",
            )
        ],
    )

    # construct domain from expected intent/entities
    domain_entities = [item[ENTITY_ATTRIBUTE_TYPE] for item in expected_entities]
    domain_intents = [expected_intent] if expected_intent is not None else []
    domain = Domain(
        intents=domain_intents,
        entities=domain_entities,
        slots=[],
        responses={},
        action_names=[],
        forms={},
    )

    # extract information
    if should_warn:
        with pytest.warns(UserWarning):
            results = regex_message_handler.process([message], domain)
    else:
        results = regex_message_handler.process([message], domain)

    assert len(results) == 1
    unpacked_message = results[0]

    assert not unpacked_message.features

    assert set(unpacked_message.data.keys()) == {
        TEXT,
        INTENT,
        INTENT_RANKING_KEY,
        ENTITIES,
    }

    assert unpacked_message.data[TEXT] == message.data[TEXT].strip()

    assert set(unpacked_message.data[INTENT].keys()) == {
        INTENT_NAME_KEY,
        PREDICTED_CONFIDENCE_KEY,
    }
    assert unpacked_message.data[INTENT][INTENT_NAME_KEY] == expected_intent
    assert (
        unpacked_message.data[INTENT][PREDICTED_CONFIDENCE_KEY] == expected_confidence
    )

    intent_ranking = unpacked_message.data[INTENT_RANKING_KEY]
    assert len(intent_ranking) == 1
    assert intent_ranking[0] == {
        INTENT_NAME_KEY: expected_intent,
        PREDICTED_CONFIDENCE_KEY: expected_confidence,
    }
    if expected_entities:
        entity_data: List[Dict[Text, Any]] = unpacked_message.data[ENTITIES]
        assert all(
            set(item.keys())
            == {
                ENTITY_ATTRIBUTE_VALUE,
                ENTITY_ATTRIBUTE_TYPE,
                ENTITY_ATTRIBUTE_START,
                ENTITY_ATTRIBUTE_END,
            }
            for item in entity_data
        )
        assert set(
            (item[ENTITY_ATTRIBUTE_TYPE], item[ENTITY_ATTRIBUTE_VALUE])
            for item in expected_entities
        ) == set(
            (item[ENTITY_ATTRIBUTE_TYPE], item[ENTITY_ATTRIBUTE_VALUE])
            for item in entity_data
        )
    else:
        assert unpacked_message.data[ENTITIES] is not None
        assert len(unpacked_message.data[ENTITIES]) == 0


@pytest.mark.parametrize(
    "intent,entities,expected_intent,domain_entities",
    [
        ("wrong_intent", {"entity": 1.0}, "other_intent", ["entity"]),
        ("my_intent", {"wrong_entity": 1.0}, "my_intent", ["other-entity"]),
        ("wrong_intent", {"wrong_entity": 1.0}, "other_intent", ["other-entity"]),
        # Special case: text "my_intent['entity1']" will be interpreted as the intent.
        # This is not caught via the regex at the moment (intent names can include
        # anything except "{" and "@".)
        ("wrong_entity", ["wrong_entity"], "wrong_entity", ["wrong_entity"]),
    ],
)
def test_process_warns_if_intent_or_entities_not_in_domain(
    regex_message_handler: RegexMessageHandlerGraphComponent,
    intent: Text,
    entities: Optional[Text],
    expected_intent: Text,
    domain_entities: List[Text],
):
    # construct text according to pattern
    text = INTENT_MESSAGE_PREFIX + intent  # do not add a confidence value
    if entities is not None:
        text += json.dumps(entities)
    message = Message(data={TEXT: text})

    # construct domain from expected intent/entities
    domain = Domain(
        intents=[expected_intent],
        entities=domain_entities,
        slots=[],
        responses={},
        action_names=[],
        forms={},
    )

    # expect a warning
    with pytest.warns(UserWarning):
        results = regex_message_handler.process([message], domain)
    unpacked_message = results[0]

    if "wrong" not in intent:
        assert unpacked_message.data[INTENT][INTENT_NAME_KEY] == intent
        if "wrong" in entities:
            assert unpacked_message.data[ENTITIES] is not None
            assert len(unpacked_message.data[ENTITIES]) == 0
    else:
        assert unpacked_message == message


@pytest.mark.parametrize(
    "text",
    [
        "some other text",
        "text" + INTENT_MESSAGE_PREFIX,
        INTENT_MESSAGE_PREFIX,
        INTENT_MESSAGE_PREFIX + "@0.5",
    ],
)
def test_process_does_not_do_anything(
    regex_message_handler: RegexMessageHandlerGraphComponent, text: Text
):

    message = Message(
        data={TEXT: text, INTENT: "bla"},
        features=[
            Features(
                features=np.zeros((1, 1)),
                feature_type=FEATURE_TYPE_SENTENCE,
                attribute=TEXT,
                origin="nlu-pipeline",
            )
        ],
    )

    # construct domain from expected intent/entities
    domain = Domain(
        intents=["intent"],
        entities=["entity"],
        slots=[],
        responses={},
        action_names=[],
        forms={},
    )

    parsed_messages = regex_message_handler.process([message], domain)

    assert parsed_messages[0] == message


async def test_correct_entity_start_and_end(
    regex_message_handler: RegexMessageHandlerGraphComponent,
):

    entity = "name"
    slot_1 = {entity: "Core"}
    text = f"/greet{json.dumps(slot_1)}"

    message = Message(data={TEXT: text},)

    domain = Domain(
        intents=["greet"],
        entities=[entity],
        slots=[],
        responses={},
        action_names=[],
        forms={},
    )

    message = regex_message_handler.process([message], domain)[0]

    assert message.data == {
        "text": '/greet{"name": "Core"}',
        "intent": {"name": "greet", "confidence": 1.0},
        "intent_ranking": [{"name": "greet", "confidence": 1.0}],
        "entities": [{"entity": "name", "value": "Core", "start": 6, "end": 22}],
    }
