from rasa.graph_components.providers.responses_provider import ResponsesProvider
from rasa.engine.graph import ExecutionContext
from rasa.engine.storage.resource import Resource
from rasa.engine.storage.storage import ModelStorage
from rasa.shared.core.domain import Domain


def test_provide(
    default_model_storage: ModelStorage, default_execution_context: ExecutionContext
):
    resource = Resource("some resource")

    domain = Domain.load(
        "data/test_from_trigger_intent_with_no_mapping_conditions/domain.yml"
    )

    provider = ResponsesProvider.create(
        {}, default_model_storage, resource, default_execution_context
    )
    responses = provider.provide(domain)

    assert responses.data == {
        "utter_greet": [{"text": "Hey! How are you?"}],
        "utter_cheer_up": [
            {
                "text": "Here is something to cheer you up:",
                "image": "https://i.imgur.com/nGF1K8f.jpg",
            }
        ],
        "utter_did_that_help": [{"text": "Did that help you?"}],
        "utter_happy": [{"text": "Great, carry on!"}],
        "utter_test_trigger": [
            {"text": "The value of test_trigger slot is: {test_trigger}"}
        ],
        "utter_goodbye": [{"text": "Bye"}],
        "utter_iamabot": [{"text": "I am a bot, powered by Rasa."}],
        "utter_ask_test_form_question1": [{"text": "test form - question 1"}],
        "utter_submit_test_form": [{"text": "Submit test form"}],
    }
