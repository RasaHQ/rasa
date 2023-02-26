from __future__ import annotations
import dataclasses

from typing import Dict, List, Text, Any

from rasa.engine.graph import GraphComponent, ExecutionContext
from rasa.engine.storage.resource import Resource
from rasa.engine.storage.storage import ModelStorage
from rasa.shared.core.domain import Domain
import rasa.shared.utils.io


@dataclasses.dataclass
class Responses:
    """Holds the responses of the domain."""

    data: Dict[Text, List[Dict[Text, Any]]]

    def fingerprint(self) -> Text:
        """Returns a fingerprint of the responses."""
        return rasa.shared.utils.io.get_dictionary_fingerprint(self.data)

    def get(self, key: Text, default: Any) -> Any:
        """Returns the value for the given key."""
        return self.data.get(key, default)


class ResponsesProvider(GraphComponent):
    """Provides responses during training and inference time."""

    @classmethod
    def create(
        cls,
        config: Dict[Text, Any],
        model_storage: ModelStorage,
        resource: Resource,
        execution_context: ExecutionContext,
    ) -> ResponsesProvider:
        """Creates component (see parent class for full docstring)."""
        return cls()

    def provide(self, domain: Domain) -> Responses:
        """Returns the responses from the given domain."""
        return Responses(data=domain.responses)
