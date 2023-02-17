from rasa.engine.graph import ExecutionContext
from rasa.engine.storage.resource import Resource
from rasa.engine.storage.storage import ModelStorage
from rasa.graph_components.providers.forms_provider import FormsProvider
from rasa.shared.core.domain import Domain


def test_provide(
    default_model_storage: ModelStorage, default_execution_context: ExecutionContext
):
    resource = Resource("some resource")

    domain = Domain.load(
        "data/test_from_trigger_intent_with_no_mapping_conditions/domain.yml"
    )

    provider = FormsProvider.create(
        {}, default_model_storage, resource, default_execution_context
    )
    forms = provider.provide(domain)

    assert forms.data == {
        "test_form": {"required_slots": ["question1"]},
        "another_form": {"required_slots": ["q2"]},
    }
