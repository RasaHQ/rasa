from __future__ import annotations
import dataclasses

from typing import Dict, Text, Any

from rasa.engine.graph import GraphComponent, ExecutionContext
from rasa.engine.storage.resource import Resource
from rasa.engine.storage.storage import ModelStorage
from rasa.shared.core.domain import Domain
import rasa.shared.utils.io


@dataclasses.dataclass
class Forms:
    """Holds the forms of the domain."""

    data: Dict[Text, Any]

    def fingerprint(self) -> Text:
        """Returns a fingerprint of the responses."""
        return rasa.shared.utils.io.get_dictionary_fingerprint(self.data)

    def get(self, key: Text, default: Any) -> Any:
        """Returns the value for the given key."""
        return self.data.get(key, default)


class FormsProvider(GraphComponent):
    """Provides forms during training and inference time."""

    @classmethod
    def create(
        cls,
        config: Dict[Text, Any],
        model_storage: ModelStorage,
        resource: Resource,
        execution_context: ExecutionContext,
    ) -> FormsProvider:
        """Creates component (see parent class for full docstring)."""
        return cls()

    def provide(self, domain: Domain) -> Forms:
        """Returns the forms from the given domain."""
        return Forms(data=domain.forms)
