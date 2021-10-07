from rasa.shared.nlu.training_data.features import Features
from typing import Text
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
    FEATURE_TYPE_SENTENCE,
    INTENT,
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
