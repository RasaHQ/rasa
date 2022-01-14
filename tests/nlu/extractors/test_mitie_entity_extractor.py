import logging

import pytest
from typing import Callable, Dict, Text, Any
import re
import copy

from _pytest.logging import LogCaptureFixture

from rasa.engine.graph import ExecutionContext
from rasa.engine.storage.resource import Resource
from rasa.engine.storage.storage import ModelStorage
from rasa.nlu.utils.mitie_utils import MitieModel, MitieNLP
from rasa.nlu.tokenizers.tokenizer import Token
from rasa.nlu.constants import EXTRACTOR, TOKENS_NAMES
from rasa.shared.nlu.training_data.message import Message
from rasa.shared.nlu.training_data.training_data import TrainingData
from rasa.shared.nlu.constants import (
    ENTITIES,
    ENTITY_ATTRIBUTE_CONFIDENCE,
    ENTITY_ATTRIBUTE_VALUE,
    ENTITY_ATTRIBUTE_START,
    ENTITY_ATTRIBUTE_END,
    ENTITY_ATTRIBUTE_TYPE,
    INTENT,
    TEXT,
)
from rasa.nlu.extractors.mitie_entity_extractor import MitieEntityExtractor


@pytest.fixture
def default_resource() -> Resource:
    return Resource("mitie")


@pytest.fixture
def mitie_model(
    default_model_storage: ModelStorage,
    default_execution_context: ExecutionContext,
    default_resource: Resource,
) -> MitieModel:
    component = MitieNLP.create(
        MitieNLP.get_default_config(),
        default_model_storage,
        default_resource,
        default_execution_context,
    )
    return component.provide()


@pytest.fixture
def create_or_load_mitie_extractor(
    default_model_storage: ModelStorage, default_execution_context: ExecutionContext
) -> Callable[[Dict[Text, Any]], MitieEntityExtractor]:
    def inner(config: Dict[Text, Any], load: bool = False) -> MitieEntityExtractor:
        if load:
            constructor = MitieEntityExtractor.load
        else:
            constructor = MitieEntityExtractor.create
        return constructor(
            model_storage=default_model_storage,
            execution_context=default_execution_context,
            resource=Resource("MitieEntityExtractor"),
            config={**MitieEntityExtractor.get_default_config(), **config},
        )

    return inner


@pytest.mark.parametrize("with_trainable_examples", [(True, False)])
def test_train_extract_load(
    create_or_load_mitie_extractor: Callable[[Dict[Text, Any]], MitieEntityExtractor],
    mitie_model: MitieModel,
    with_trainable_examples: bool,
):

    # some texts where last token is a city
    texts_ending_with_city = ["Bert lives in Berlin", "Ernie asks where is Bielefeld"]

    # create some messages with entities
    messages_with_entities = []
    for text in texts_ending_with_city:
        tokens = [
            Token(text=match.group(), start=match.start(), end=match.end())
            for match in re.finditer(r"\w+", text)
        ]
        entities = [
            {
                ENTITY_ATTRIBUTE_TYPE: "city",
                ENTITY_ATTRIBUTE_VALUE: tokens[-1].text,
                ENTITY_ATTRIBUTE_START: tokens[-1].start,
                ENTITY_ATTRIBUTE_END: tokens[-1].end,
                EXTRACTOR: None,  # must be None or mitie_entity_extractor.name
            }
        ]

        message = Message(text=text)
        message.data[TOKENS_NAMES[TEXT]] = tokens
        message.data[ENTITIES] = entities
        if with_trainable_examples:
            message.data[INTENT] = "must have intent otherwise not an NLU example"
        else:
            pass  # not adding an intent is sufficient to make this a "core example"
        messages_with_entities.append(message)

    # turn them into training data
    training_data = TrainingData(messages_with_entities)

    # train the extractor
    mitie_entity_extractor = create_or_load_mitie_extractor(config={}, load=False)
    mitie_entity_extractor.train(training_data, model=mitie_model)

    # create some messages "without entities" - for processing
    messages_without_entities = [
        Message(
            data={
                TEXT: message.data[TEXT],
                TOKENS_NAMES[TEXT]: message.data[TOKENS_NAMES[TEXT]],
            }
        )
        for message in messages_with_entities
    ]

    # process!
    mitie_entity_extractor.process(
        messages=messages_without_entities, model=mitie_model
    )

    # check that extractor added the expected entities to the messages
    # (that initially were) "with no entities"
    if with_trainable_examples:
        for processed_message, labeled_message in zip(
            messages_without_entities, messages_with_entities
        ):  # i.e. "without (before process)"
            assert ENTITIES in processed_message.data
            computed_entities = processed_message.data[ENTITIES]
            assert len(computed_entities) == 1
            computed_entity = copy.copy(computed_entities[0])  # we need it later
            # check confidence
            assert computed_entity.pop(ENTITY_ATTRIBUTE_CONFIDENCE, "surprise") is None
            # check extractor
            assert computed_entity.pop(EXTRACTOR, None) == mitie_entity_extractor.name
            # compare the rest
            expected_entity = labeled_message.data[ENTITIES][0]
            expected_entity.pop(EXTRACTOR)
            assert computed_entity == expected_entity

    else:
        for processed_message in messages_without_entities:
            assert ENTITIES not in processed_message.data

    # load the same extractor again
    loaded_extractor = create_or_load_mitie_extractor(config={}, load=True)

    # check results are the same
    same_messages_without_entities = [
        Message(
            data={
                TEXT: message.data[TEXT],
                TOKENS_NAMES[TEXT]: message.data[TOKENS_NAMES[TEXT]],
            }
        )
        for message in messages_with_entities
    ]
    loaded_extractor.process(messages=same_messages_without_entities, model=mitie_model)
    assert same_messages_without_entities[0].data == messages_without_entities[0].data


def test_load_without_training(
    create_or_load_mitie_extractor: Callable[[Dict[Text, Any]], MitieEntityExtractor],
    caplog: LogCaptureFixture,
):
    with caplog.at_level(logging.DEBUG):
        create_or_load_mitie_extractor({}, load=True)

    assert any(
        "Failed to load MitieEntityExtractor from model storage." in message
        for message in caplog.messages
    )
