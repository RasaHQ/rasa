import copy
from rasa.engine.graph import ExecutionContext
from rasa.engine.storage.storage import ModelStorage
from typing import Any, Text, Dict, List, Callable

import pytest

from rasa.engine.storage.resource import Resource
from rasa.shared.nlu.training_data.training_data import TrainingData
from rasa.shared.nlu.training_data.message import Message
from rasa.shared.nlu.constants import (
    ENTITIES,
    ENTITY_ATTRIBUTE_TYPE,
    ENTITY_ATTRIBUTE_VALUE,
    ENTITY_ATTRIBUTE_START,
    ENTITY_ATTRIBUTE_END,
    TEXT,
    INTENT,
    EXTRACTOR,
)
from rasa.nlu.extractors.regex_entity_extractor import RegexEntityExtractor


@pytest.fixture()
def create_or_load_extractor(
    default_model_storage: ModelStorage, default_execution_context: ExecutionContext
) -> Callable[..., RegexEntityExtractor]:
    def inner(config: Dict[Text, Any], load: bool = False) -> RegexEntityExtractor:
        if load:
            constructor = RegexEntityExtractor.load
        else:
            constructor = RegexEntityExtractor.create
        return constructor(
            config=config,
            model_storage=default_model_storage,
            resource=Resource("regex"),
            execution_context=default_execution_context,
        )

    return inner


@pytest.mark.parametrize(
    "config, text, lookup, expected_entities, test_loading",
    [
        (
            # default config
            {},
            "Berlin and London are cities.",
            [
                {
                    "name": "city",
                    "elements": ["Berlin", "Amsterdam", "New York", "London"],
                }
            ],
            [
                {
                    ENTITY_ATTRIBUTE_TYPE: "city",
                    ENTITY_ATTRIBUTE_VALUE: "Berlin",
                    ENTITY_ATTRIBUTE_START: 0,
                    ENTITY_ATTRIBUTE_END: 6,
                    EXTRACTOR: RegexEntityExtractor.__name__,
                },
                {
                    ENTITY_ATTRIBUTE_TYPE: "city",
                    ENTITY_ATTRIBUTE_VALUE: "London",
                    ENTITY_ATTRIBUTE_START: 11,
                    ENTITY_ATTRIBUTE_END: 17,
                    EXTRACTOR: RegexEntityExtractor.__name__,
                },
            ],
            True,  # test loading
        ),
        (
            {},
            "Sophie is visiting Thomas in Berlin.",
            [
                {
                    "name": "city",
                    "elements": ["Berlin", "Amsterdam", "New York", "London"],
                },
                {"name": "person", "elements": ["Max", "John", "Sophie", "Lisa"]},
            ],
            [
                {
                    ENTITY_ATTRIBUTE_TYPE: "city",
                    ENTITY_ATTRIBUTE_VALUE: "Berlin",
                    ENTITY_ATTRIBUTE_START: 29,
                    ENTITY_ATTRIBUTE_END: 35,
                    EXTRACTOR: RegexEntityExtractor.__name__,
                },
                {
                    ENTITY_ATTRIBUTE_TYPE: "person",
                    ENTITY_ATTRIBUTE_VALUE: "Sophie",
                    ENTITY_ATTRIBUTE_START: 0,
                    ENTITY_ATTRIBUTE_END: 6,
                    EXTRACTOR: RegexEntityExtractor.__name__,
                },
            ],
            False,
        ),
        (
            {},
            "Rasa is great.",
            [
                {
                    "name": "city",
                    "elements": ["Berlin", "Amsterdam", "New York", "London"],
                },
                {"name": "person", "elements": ["Max", "John", "Sophie", "Lisa"]},
            ],
            [],
            False,
        ),
        # not using word boundaries
        (
            {"use_word_boundaries": False},
            "北京和上海都是大城市。",
            [{"name": "city", "elements": ["北京", "上海", "广州", "深圳", "杭州"]}],
            [
                {
                    ENTITY_ATTRIBUTE_TYPE: "city",
                    ENTITY_ATTRIBUTE_VALUE: "北京",
                    ENTITY_ATTRIBUTE_START: 0,
                    ENTITY_ATTRIBUTE_END: 2,
                    EXTRACTOR: RegexEntityExtractor.__name__,
                },
                {
                    ENTITY_ATTRIBUTE_TYPE: "city",
                    ENTITY_ATTRIBUTE_VALUE: "上海",
                    ENTITY_ATTRIBUTE_START: 3,
                    ENTITY_ATTRIBUTE_END: 5,
                    EXTRACTOR: RegexEntityExtractor.__name__,
                },
            ],
            True,  # test loading
        ),
        (
            {"use_word_boundaries": False},
            "小明正要去北京拜访老李。",
            [
                {"name": "city", "elements": ["北京", "上海", "广州", "深圳", "杭州"]},
                {"name": "person", "elements": ["小明", "小红", "小王", "小李"]},
            ],
            [
                {
                    ENTITY_ATTRIBUTE_TYPE: "city",
                    ENTITY_ATTRIBUTE_VALUE: "北京",
                    ENTITY_ATTRIBUTE_START: 5,
                    ENTITY_ATTRIBUTE_END: 7,
                    EXTRACTOR: RegexEntityExtractor.__name__,
                },
                {
                    ENTITY_ATTRIBUTE_TYPE: "person",
                    ENTITY_ATTRIBUTE_VALUE: "小明",
                    ENTITY_ATTRIBUTE_START: 0,
                    ENTITY_ATTRIBUTE_END: 2,
                    EXTRACTOR: RegexEntityExtractor.__name__,
                },
            ],
            True,
        ),
        (
            {"use_word_boundaries": False},
            "Rasa 真好用。",
            [
                {"name": "city", "elements": ["北京", "上海", "广州", "深圳", "杭州"]},
                {"name": "person", "elements": ["小明", "小红", "小王", "小李"]},
            ],
            [],
            False,
        ),
        # case sensitivity
        (
            {"case_sensitive": True},
            "berlin and London are cities.",
            [
                {
                    "name": "city",
                    "elements": ["Berlin", "Amsterdam", "New York", "London"],
                }
            ],
            [
                {
                    ENTITY_ATTRIBUTE_TYPE: "city",
                    ENTITY_ATTRIBUTE_VALUE: "London",
                    ENTITY_ATTRIBUTE_START: 11,
                    ENTITY_ATTRIBUTE_END: 17,
                    EXTRACTOR: RegexEntityExtractor.__name__,
                }
            ],
            True,
        ),
        (
            {"case_sensitive": False},
            "berlin and London are cities.",
            [
                {
                    "name": "city",
                    "elements": ["Berlin", "Amsterdam", "New York", "london"],
                }
            ],
            [
                {
                    ENTITY_ATTRIBUTE_TYPE: "city",
                    ENTITY_ATTRIBUTE_VALUE: "berlin",
                    ENTITY_ATTRIBUTE_START: 0,
                    ENTITY_ATTRIBUTE_END: 6,
                    EXTRACTOR: RegexEntityExtractor.__name__,
                },
                {
                    ENTITY_ATTRIBUTE_TYPE: "city",
                    ENTITY_ATTRIBUTE_VALUE: "London",
                    ENTITY_ATTRIBUTE_START: 11,
                    ENTITY_ATTRIBUTE_END: 17,
                    EXTRACTOR: RegexEntityExtractor.__name__,
                },
            ],
            False,
        ),
    ],
)
def test_train_and_process(
    create_or_load_extractor: Callable[..., RegexEntityExtractor],
    config: Dict[Text, Any],
    text: Text,
    lookup: List[Dict[Text, List[Text]]],
    expected_entities: List[Dict[Text, Any]],
    test_loading: bool,
):
    message = Message(data={TEXT: text})
    if test_loading:
        message_copy = copy.deepcopy(message)

    training_data = TrainingData()
    training_data.lookup_tables = lookup
    training_data.training_examples = [
        Message(
            data={
                TEXT: "Hi Max!",
                INTENT: "greet",
                ENTITIES: [{"entity": "person", "value": "Max"}],
            }
        ),
        Message(
            data={
                TEXT: "I live in Berlin",
                INTENT: "inform",
                ENTITIES: [{"entity": "city", "value": "Berlin"}],
            }
        ),
    ]

    entity_extractor = create_or_load_extractor(config)
    entity_extractor.train(training_data)
    entity_extractor.process([message])
    entities = message.get(ENTITIES)
    assert entities == expected_entities

    if test_loading:
        loaded_entity_extractor = create_or_load_extractor(config, load=True)
        loaded_entity_extractor.process([message_copy])
        loaded_entity_extractor.patterns == entity_extractor.patterns


def test_train_process_and_load_with_empty_model(
    create_or_load_extractor: Callable[..., RegexEntityExtractor]
):
    extractor = create_or_load_extractor({})
    with pytest.warns(UserWarning):
        extractor.train(TrainingData([]))
    with pytest.warns(UserWarning):
        extractor.process(Message(data={TEXT: "arbitrary"}))
    with pytest.warns(UserWarning):
        create_or_load_extractor({}, load=True)


def test_process_does_not_overwrite_any_entities(
    create_or_load_extractor: Callable[..., RegexEntityExtractor]
):

    pre_existing_entity = {
        ENTITY_ATTRIBUTE_TYPE: "person",
        ENTITY_ATTRIBUTE_VALUE: "Max",
        ENTITY_ATTRIBUTE_START: 0,
        ENTITY_ATTRIBUTE_END: 3,
        EXTRACTOR: "other extractor",
    }
    message = Message(data={TEXT: "Max lives in Berlin.", INTENT: "infrom"})
    message.set(ENTITIES, [copy.deepcopy(pre_existing_entity)])

    training_data = TrainingData()
    training_data.training_examples = [
        Message(
            data={
                TEXT: "Hi Max!",
                INTENT: "greet",
                ENTITIES: [
                    {ENTITY_ATTRIBUTE_TYPE: "person", ENTITY_ATTRIBUTE_VALUE: "Max"}
                ],
            }
        ),
        Message(
            data={
                TEXT: "I live in Berlin",
                INTENT: "inform",
                ENTITIES: [
                    {ENTITY_ATTRIBUTE_TYPE: "city", ENTITY_ATTRIBUTE_VALUE: "Berlin"}
                ],
            }
        ),
    ]
    training_data.lookup_tables = [
        {"name": "city", "elements": ["London", "Berlin", "Amsterdam"]}
    ]

    entity_extractor = create_or_load_extractor(config={})
    entity_extractor.train(training_data)
    entity_extractor.process([message])

    entities = message.get(ENTITIES)
    assert entities == [
        pre_existing_entity,
        {
            ENTITY_ATTRIBUTE_TYPE: "city",
            ENTITY_ATTRIBUTE_VALUE: "Berlin",
            ENTITY_ATTRIBUTE_START: 13,
            ENTITY_ATTRIBUTE_END: 19,
            EXTRACTOR: RegexEntityExtractor.__name__,
        },
    ]
