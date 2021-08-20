from typing import Dict, Text, Any, Optional

import pytest

from rasa.core.channels import UserMessage
from rasa.engine.graph import ExecutionContext
from rasa.engine.storage.resource import Resource
from rasa.engine.storage.storage import ModelStorage
from rasa.graph_components.converters.nlu_message_converter import NLUMessageConverter
from rasa.shared.nlu.training_data.message import Message


@pytest.mark.parametrize(
    "config, text",
    [
        ({}, "Hi"),
        ({"augmentation_factor": 0}, "Hi"),
        ({"use_story_concatenation": False}, "Hi"),
        (
            {
                "remove_duplicates": True,
                "unique_last_num_states": None,
                "augmentation_factor": 50,
                "tracker_limit": None,
                "use_story_concatenation": True,
                "debug_plots": False,
            },
            "Hi",
        ),
        ({}, None),
        ({"augmentation_factor": 0}, None),
        ({"use_story_concatenation": False}, None),
        (
            {
                "remove_duplicates": True,
                "unique_last_num_states": None,
                "augmentation_factor": 50,
                "tracker_limit": None,
                "use_story_concatenation": True,
                "debug_plots": False,
            },
            None,
        ),
    ],
)
def test_nlu_message_converter_converts_message(
    default_model_storage: ModelStorage,
    default_execution_context: ExecutionContext,
    config: Dict[Text, Any],
    text: Optional[Text],
):
    component = NLUMessageConverter.create(
        {**NLUMessageConverter.get_default_config(), **config},
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
