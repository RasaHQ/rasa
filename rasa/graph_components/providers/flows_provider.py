from __future__ import annotations

from typing import Any, Dict, Text

from rasa.engine.graph import ExecutionContext, GraphComponent
from rasa.engine.storage.resource import Resource
from rasa.engine.storage.storage import ModelStorage
from rasa.shared.core.flows import FlowsList
from rasa.shared.core.flows.yaml_flows_io import YAMLFlowsReader, YamlFlowsWriter
from rasa.shared.importers.importer import TrainingDataImporter

FLOWS_PERSISTENCE_FILE_NAME = "flows.yml"


class FlowsProvider(GraphComponent):
    """Provides flows information during training and inference time."""

    def __init__(
        self,
        model_storage: ModelStorage,
        resource: Resource,
        flows: FlowsList,
    ) -> None:
        """Creates flows provider."""
        self._model_storage = model_storage
        self._resource = resource
        self._flows = flows

    @classmethod
    def create(
        cls,
        config: Dict[Text, Any],
        model_storage: ModelStorage,
        resource: Resource,
        execution_context: ExecutionContext,
    ) -> FlowsProvider:
        """Creates component (see parent class for full docstring)."""
        return cls(model_storage, resource, flows=FlowsList(underlying_flows=[]))

    @classmethod
    def load(
        cls,
        config: Dict[Text, Any],
        model_storage: ModelStorage,
        resource: Resource,
        execution_context: ExecutionContext,
        **kwargs: Any,
    ) -> FlowsProvider:
        """Creates provider using a persisted version of itself."""
        with model_storage.read_from(resource) as resource_directory:
            flows = YAMLFlowsReader.read_from_file(
                resource_directory / FLOWS_PERSISTENCE_FILE_NAME
            )
        flows.validate()
        return cls(model_storage, resource, flows)

    def _persist(self, flows: FlowsList) -> None:
        """Persists flows to model storage."""
        with self._model_storage.write_to(self._resource) as resource_directory:
            YamlFlowsWriter.dump(
                flows.underlying_flows,
                resource_directory / FLOWS_PERSISTENCE_FILE_NAME,
            )

    def provide_train(self, importer: TrainingDataImporter) -> FlowsList:
        """Provides flows configuration from training data during training."""
        self._flows = importer.get_flows()
        self._persist(self._flows)
        return self._flows

    def provide_inference(self) -> FlowsList:
        """Provides the flows configuration during inference."""
        return self._flows
