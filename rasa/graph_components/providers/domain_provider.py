from __future__ import annotations
from typing import Dict, Text, Any, Optional

from rasa.engine.graph import GraphComponent, ExecutionContext
from rasa.engine.storage.resource import Resource
from rasa.engine.storage.storage import ModelStorage
from rasa.shared.core.domain import Domain
from rasa.shared.importers.importer import TrainingDataImporter


class DomainProvider(GraphComponent):
    """Provides domain during training and inference time."""

    def __init__(
        self,
        model_storage: ModelStorage,
        resource: Resource,
        domain: Optional[Domain] = None,
    ) -> None:
        """Creates domain provider."""
        self._model_storage = model_storage
        self._resource = resource
        self.domain = domain

    @classmethod
    def create(
        cls,
        config: Dict[Text, Any],
        model_storage: ModelStorage,
        resource: Resource,
        execution_context: ExecutionContext,
    ) -> DomainProvider:
        """Creates component (see parent class for full docstring)."""
        with model_storage.read_from(resource) as resource_directory:
            domain = Domain.from_path(resource_directory)
        return cls(model_storage, resource, domain)

    def persist(self, domain: Domain) -> None:
        """Persists domain to model storage."""
        with self._model_storage.write_to(self._resource) as resource_directory:
            domain.persist(resource_directory.joinpath("domain.yml"))

    def provide_train(self, importer: TrainingDataImporter) -> Domain:
        """Generates loaded Domain of the bot."""
        domain = importer.get_domain()
        self.persist(domain)
        return domain

    def provide_inference(self):
        """Provides the domain during inference."""
        return self.domain
