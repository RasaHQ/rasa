from __future__ import annotations

import dataclasses
import logging
from typing import Any, Dict, Text

import rasa.shared.utils.io
from rasa.engine.graph import ExecutionContext, GraphComponent
from rasa.engine.storage.resource import Resource
from rasa.engine.storage.storage import ModelStorage

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class RuleOnlyDataProvider(GraphComponent):
    """Provides slots and loops that are only used in rules to other policies.

    Policies can use this to exclude features which are only used by rules from the
    featurization.
    """

    rule_only_data: Dict[Text, Any]

    @classmethod
    def create(
        cls,
        config: Dict[Text, Any],
        model_storage: ModelStorage,
        resource: Resource,
        execution_context: ExecutionContext,
    ) -> RuleOnlyDataProvider:
        """Creates component (see parent class for docstring)."""
        rule_only_data = {}
        try:
            with model_storage.read_from(resource) as directory:
                rule_only_data = rasa.shared.utils.io.read_json_file(
                    directory / "rule_only_data.json"
                )
        except ValueError:
            logger.debug(
                "Failed to load rule-only data from a trained 'RulePolicy'. "
                "Providing empty rule-only data instead."
            )

        return cls(rule_only_data)

    def provide(self) -> Dict[Text, Any]:
        """Provides data to other graph component."""
        return self.rule_only_data
