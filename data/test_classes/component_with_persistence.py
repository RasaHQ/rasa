from __future__ import annotations
import json
from typing import Dict, Text, Any, Optional

from rasa.engine.graph import GraphComponent, ExecutionContext
from rasa.engine.storage.resource import Resource
from rasa.engine.storage.storage import ModelStorage
from rasa.shared.nlu.training_data.training_data import TrainingData


class MyComponent(GraphComponent):
    def __init__(
        self,
        model_storage: ModelStorage,
        resource: Resource,
        training_artifact: Optional[Dict],
    ) -> None:
        self._model_storage = model_storage
        self._resource = resource

    @classmethod
    def create(
        cls,
        config: Dict[Text, Any],
        model_storage: ModelStorage,
        resource: Resource,
        execution_context: ExecutionContext,
    ) -> MyComponent:
        return cls(model_storage, resource, training_artifact=None)

    def train(self, training_data: TrainingData) -> Resource:
        # Train your component
        ...

        # Persist your component
        with self._model_storage.write_to(self._resource) as directory_path:
            with open(directory_path / "artifact.json", "w") as file:
                json.dump({"my": "training artifact"}, file)

        # Return resource to make sure the training artifacts
        # can be cached.
        return self._resource

    @classmethod
    def load(
        cls,
        config: Dict[Text, Any],
        model_storage: ModelStorage,
        resource: Resource,
        execution_context: ExecutionContext,
        **kwargs: Any,
    ) -> MyComponent:
        try:
            with model_storage.read_from(resource) as directory_path:
                with open(directory_path / "artifact.json", "r") as file:
                    training_artifact = json.load(file)
                    return cls(
                        model_storage, resource, training_artifact=training_artifact
                    )
        except ValueError:
            # This allows you to handle the case if there was no
            # persisted data for your component
            ...
