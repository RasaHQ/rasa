from typing import Text

import pytest
from rasa.engine.graph import ExecutionContext
from rasa.engine.storage.resource import Resource
from rasa.engine.storage.storage import ModelStorage
from rasa.graph_components.providers.domain_provider import DomainProvider
from rasa.shared.core.domain import Domain
from rasa.shared.exceptions import InvalidConfigException
from rasa.shared.importers.importer import TrainingDataImporter


def test_domain_provider_provides_and_persists_domain(
    default_model_storage: ModelStorage,
    default_execution_context: ExecutionContext,
    config_path: Text,
    domain_path: Text,
    domain: Domain,
):
    resource = Resource("xy")
    component = DomainProvider.create(
        DomainProvider.get_default_config(),
        default_model_storage,
        resource,
        default_execution_context,
    )
    assert isinstance(component, DomainProvider)

    importer = TrainingDataImporter.load_from_config(config_path, domain_path)
    training_domain = component.provide_train(importer)

    assert isinstance(training_domain, Domain)
    assert domain.fingerprint() == training_domain.fingerprint()

    with default_model_storage.read_from(resource) as d:
        match = list(d.glob("**/domain.yml"))
        assert len(match) == 1
        assert match[0].is_file()
        assert domain.fingerprint() == Domain.from_path(match[0]).fingerprint()

    component_2 = DomainProvider.load(
        {}, default_model_storage, resource, default_execution_context
    )
    inference_domain = component_2.provide_inference()

    assert isinstance(inference_domain, Domain)
    assert domain.fingerprint() == inference_domain.fingerprint()


def test_provide_without_domain(
    default_model_storage: ModelStorage, default_execution_context: ExecutionContext
):
    component = DomainProvider.create(
        DomainProvider.get_default_config(),
        default_model_storage,
        Resource("some resource"),
        default_execution_context,
    )

    with pytest.raises(InvalidConfigException):
        component.provide_inference()
