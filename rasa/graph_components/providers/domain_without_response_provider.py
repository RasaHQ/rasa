from __future__ import annotations
from typing import Dict, Text, Any
import copy

from rasa.engine.graph import GraphComponent, ExecutionContext
from rasa.engine.storage.resource import Resource
from rasa.engine.storage.storage import ModelStorage
from rasa.shared.core.domain import Domain


class DomainWithoutResponseProvider(GraphComponent):
    """Provides domain without information about responses."""

    @classmethod
    def create(
        cls,
        config: Dict[Text, Any],
        model_storage: ModelStorage,
        resource: Resource,
        execution_context: ExecutionContext,
    ) -> DomainWithoutResponseProvider:
        """Creates component (see parent class for full docstring)."""
        return cls()

    def recreate_without_responses(self, domain: Domain) -> Domain:
        """Recreates the given domain but acts as if responses have not been specified.

        Args:
            domain: A domain.

        Returns:
            Domain that has been created from the same parameters as the given domain
            but with an empty set of responses.
        """
        responses = dict()
        domain = copy.deepcopy(domain)
        domain.responses = responses
        domain.user_actions = domain._custom_actions
        domain.action_names_or_texts = (
            domain._combine_user_with_default_actions(domain.user_actions)
            + [
                form_name
                for form_name in domain.form_names
                if form_name not in domain._custom_actions
            ]
            + domain.action_texts
        )

        return domain
