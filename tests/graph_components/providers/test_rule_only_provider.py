import rasa.core.training
from rasa.core.policies.rule_policy import RulePolicy
from rasa.engine.graph import ExecutionContext
from rasa.engine.storage.resource import Resource
from rasa.engine.storage.storage import ModelStorage
from rasa.graph_components.providers.rule_only_provider import RuleOnlyDataProvider
from rasa.shared.core.constants import RULE_ONLY_SLOTS, RULE_ONLY_LOOPS
from rasa.shared.core.domain import Domain


def test_provide(
    default_model_storage: ModelStorage, default_execution_context: ExecutionContext
):
    resource = Resource("some resource")

    domain = Domain.load("examples/rules/domain.yml")
    trackers = rasa.core.training.load_data("examples/rules/data/rules.yml", domain)

    policy = RulePolicy.create(
        RulePolicy.get_default_config(),
        default_model_storage,
        resource,
        default_execution_context,
    )

    policy.train(trackers, domain)

    provider = RuleOnlyDataProvider.load(
        {}, default_model_storage, resource, default_execution_context
    )
    rule_only_data = provider.provide()

    assert rule_only_data

    for key in [RULE_ONLY_SLOTS, RULE_ONLY_LOOPS]:
        assert rule_only_data[key] == policy.lookup[key]
