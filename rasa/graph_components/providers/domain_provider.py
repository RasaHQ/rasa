from __future__ import annotations

from typing import Any, Dict, Optional, Text

from rasa.engine.graph import ExecutionContext, GraphComponent
from rasa.engine.storage.resource import Resource
from rasa.engine.storage.storage import ModelStorage
from rasa.shared.core.domain import Domain
from rasa.shared.exceptions import InvalidConfigException
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
        self._domain = domain

    @classmethod
    def create(
        cls,
        config: Dict[Text, Any],
        model_storage: ModelStorage,
        resource: Resource,
        execution_context: ExecutionContext,
    ) -> DomainProvider:
        """Creates component (see parent class for full docstring)."""
        return cls(model_storage, resource)

    @classmethod
    def load(
        cls,
        config: Dict[Text, Any],
        model_storage: ModelStorage,
        resource: Resource,
        execution_context: ExecutionContext,
        **kwargs: Any,
    ) -> DomainProvider:
        """Creates provider using a persisted version of itself."""
        with model_storage.read_from(resource) as resource_directory:
            domain = Domain.from_path(resource_directory)
        return cls(model_storage, resource, domain)

    def _persist(self, domain: Domain) -> None:
        """Persists domain to model storage."""
        with self._model_storage.write_to(self._resource) as resource_directory:
            domain.persist(resource_directory / "domain.yml")

    def provide_train(self, importer: TrainingDataImporter) -> Domain:
        """Provides domain from training data during training."""
        domain = importer.get_domain()
        self._persist(domain)
        return domain

    def provide_inference(self) -> Domain:
        """Provides the domain during inference."""
        if self._domain is None:
            # This can't really happen but if it happens then we fail early
            raise InvalidConfigException(
                "No domain was found. This is required for "
                "making model predictions. Please make sure to "
                "provide a valid domain during training."
            )
        return self._domain
