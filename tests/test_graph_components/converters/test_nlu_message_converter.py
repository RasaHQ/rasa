from typing import Text, Optional

import pytest

from rasa.core.channels import UserMessage
from rasa.engine.graph import ExecutionContext
from rasa.engine.storage.resource import Resource
from rasa.engine.storage.storage import ModelStorage
from rasa.graph_components.converters.nlu_message_converter import NLUMessageConverter
from rasa.shared.nlu.training_data.message import Message


@pytest.mark.parametrize(
    "text", ["Hi", None],
)
def test_nlu_message_converter_converts_message(
    default_model_storage: ModelStorage,
    default_execution_context: ExecutionContext,
    text: Optional[Text],
):
    component = NLUMessageConverter.create(
        {**NLUMessageConverter.get_default_config()},
        default_model_storage,
        Resource("test"),
        default_execution_context,
    )

    if text:
        message = UserMessage(text=text)
        nlu_message = component.convert_user_message(message)
        assert len(nlu_message) == 1
        assert isinstance(nlu_message[0], Message)

        assert nlu_message[0].get("text") == "Hi"
        assert nlu_message[0].get("metadata") is None
    else:
        message = None
        nlu_message = component.convert_user_message(message)
        assert len(nlu_message) == 0
