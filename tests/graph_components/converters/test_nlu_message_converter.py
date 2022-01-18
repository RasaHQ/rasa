from rasa.core.channels import UserMessage
from rasa.engine.graph import ExecutionContext
from rasa.engine.storage.resource import Resource
from rasa.engine.storage.storage import ModelStorage
from rasa.graph_components.converters.nlu_message_converter import NLUMessageConverter
from rasa.shared.nlu.constants import TEXT, TEXT_TOKENS
from rasa.shared.nlu.training_data.message import Message


def test_nlu_message_converter_converts_message(
    default_model_storage: ModelStorage, default_execution_context: ExecutionContext
):
    component = NLUMessageConverter.create(
        {**NLUMessageConverter.get_default_config()},
        default_model_storage,
        Resource("test"),
        default_execution_context,
    )

    message = UserMessage(text="Hello", metadata=None)
    nlu_message = component.convert_user_message([message])
    assert len(nlu_message) == 1
    assert isinstance(nlu_message[0], Message)

    assert nlu_message[0].get("text") == "Hello"
    assert nlu_message[0].get("metadata") is None
    assert nlu_message[0].output_properties == {TEXT_TOKENS, TEXT}


def test_nlu_message_converter_converts_message_with_metadata(
    default_model_storage: ModelStorage, default_execution_context: ExecutionContext
):
    component = NLUMessageConverter.create(
        {}, default_model_storage, Resource("with_metadata"), default_execution_context
    )

    message = UserMessage(text="Hello", metadata={"test_key": "test_value"})
    nlu_message = component.convert_user_message([message])
    assert len(nlu_message) == 1
    assert isinstance(nlu_message[0], Message)

    assert nlu_message[0].get("text") == "Hello"
    assert nlu_message[0].get("metadata") == {"test_key": "test_value"}


def test_nlu_message_converter_handles_no_user_message(
    default_model_storage: ModelStorage, default_execution_context: ExecutionContext
):
    component = NLUMessageConverter.create(
        {},
        default_model_storage,
        Resource("no_user_message"),
        default_execution_context,
    )

    nlu_message = component.convert_user_message([])
    assert len(nlu_message) == 0
