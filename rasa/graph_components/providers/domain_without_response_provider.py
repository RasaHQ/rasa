from __future__ import annotations

import copy
from typing import Dict, Text, Any

from rasa.engine.graph import GraphComponent, ExecutionContext
from rasa.engine.storage.resource import Resource
from rasa.engine.storage.storage import ModelStorage
from rasa.shared.core.domain import KEY_RESPONSES, Domain


class DomainWithoutResponsesProvider(GraphComponent):
    """Provides domain without information about responses."""

    @classmethod
    def create(
        cls,
        config: Dict[Text, Any],
        model_storage: ModelStorage,
        resource: Resource,
        execution_context: ExecutionContext,
    ) -> DomainWithoutResponsesProvider:
        """Creates component (see parent class for full docstring)."""
        return cls()

    def provide(self, domain: Domain) -> Domain:
        """Recreates the given domain but acts as if responses have not been specified.

        Args:
            domain: A domain.

        Returns:
            Domain that has been created from the same parameters as the given domain
            but with an empty set of responses.
        """
        serialized_domain = copy.deepcopy(domain.as_dict())

        for response_name in serialized_domain[KEY_RESPONSES]:
            serialized_domain[KEY_RESPONSES][response_name] = []

        return Domain.from_dict(serialized_domain)
