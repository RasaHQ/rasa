from __future__ import annotations

import copy
from typing import Any, Dict, Text

from rasa.engine.graph import ExecutionContext, GraphComponent
from rasa.engine.storage.resource import Resource
from rasa.engine.storage.storage import ModelStorage
from rasa.shared.constants import REQUIRED_SLOTS_KEY
from rasa.shared.core.domain import KEY_FORMS, KEY_RESPONSES, SESSION_CONFIG_KEY, Domain


class DomainForCoreTrainingProvider(GraphComponent):
    """Provides domain without information that is irrelevant for core training.

    The information that we retain includes:
    - intents and their "used" and "ignored" entities because intents influence the
      next action prediction directly and the latter flags determine whether the
      listed entities influence the next action prediction
    - entities, their roles and groups, and their `influence_conversation` flag because
      all of those items are used by policies
    - slots names along with their types, since this type information determines the
      pre-featurization of slot values
    - response keys (i.e. `utter_*) because those keys may appear in stories
    - form names because those appear in stories
    - how slots are filled (i.e. 'mappings' key under 'slots') because a domain instance
      needs to be created by core during training time to parse the training data
      properly

    This information that we drop (or replace with default values) includes:
    - the 'session_config' which determines details of a session e.g. whether data is
      transferred from one session to the next (this is replaced with defaults as it
      cannot just be removed)
    - the actual text of a 'response' because those are only used by response selectors
    - the actual configuration of 'forms' because those are not actually executed
      by core components

    References:
        - `rasa.core.featurizer.tracker_featurizer.py` (used by all policies)
        - `rasa.core.featurizer.single_state_featurizer.py` (used by ML policies)
        - `rasa.shared.core.domain.get_active_state` (used by above references)
        - `rasa.shared.core.slots.as_features` (used by above references)
        - `rasa.shared.core.training_data.structures.StoryStep.explicit_events`
           (i.e. slots needed for core training)
    """

    @classmethod
    def create(
        cls,
        config: Dict[Text, Any],
        model_storage: ModelStorage,
        resource: Resource,
        execution_context: ExecutionContext,
    ) -> DomainForCoreTrainingProvider:
        """Creates component (see parent class for full docstring)."""
        return cls()

    def provide(self, domain: Domain) -> Domain:
        """Recreates the given domain but drops information that is irrelevant for core.

        Args:
            domain: A domain.

        Returns:
            A similar domain without information that is irrelevant for core training.
        """
        return self.create_pruned_version(domain)

    @staticmethod
    def create_pruned_version(domain: Domain) -> Domain:
        """Recreates the given domain but drops information that is irrelevant for core.

        Args:
            domain: A domain.

        Returns:
             A similar domain without information that is irrelevant for core training.
        """
        serialized_domain = copy.deepcopy(domain.as_dict())

        serialized_domain.pop("config", None)  # `store_entities_as_slots`
        serialized_domain.pop(SESSION_CONFIG_KEY, None)
        for response_name in serialized_domain.get(KEY_RESPONSES, []):
            serialized_domain[KEY_RESPONSES][response_name] = []
        for form_name in serialized_domain.get(KEY_FORMS, []):
            serialized_domain[KEY_FORMS][form_name] = {REQUIRED_SLOTS_KEY: []}
        return Domain.from_dict(serialized_domain)
