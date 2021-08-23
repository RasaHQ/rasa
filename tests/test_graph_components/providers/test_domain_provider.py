from pathlib import Path
from typing import Text

from rasa.engine.graph import ExecutionContext
from rasa.engine.storage.resource import Resource
from rasa.engine.storage.storage import ModelStorage
from rasa.graph_components.providers.domain_provider import DomainProvider
from rasa.shared.core.domain import Domain
from rasa.shared.importers.importer import TrainingDataImporter


def test_domain_provider_generates_and_persists_domain(
    default_model_storage: ModelStorage,
    default_execution_context: ExecutionContext,
    tmp_path: Path,
    config_path: Text,
    domain_path: Text,
):
    resource = Resource.from_cache("xy", tmp_path, default_model_storage)
    component = DomainProvider.create(
        DomainProvider.get_default_config(),
        default_model_storage,
        resource,
        default_execution_context,
    )
    assert isinstance(component, DomainProvider)

    importer = TrainingDataImporter.load_from_config(config_path, domain_path)
    domain = component.provide_train(importer)

    assert isinstance(domain, Domain)

    with default_model_storage.read_from(resource) as d:
        match = list(d.glob("**/domain.yml"))
        assert match[0].is_file()
        assert domain.fingerprint() == Domain.from_path(match[0]).fingerprint()
