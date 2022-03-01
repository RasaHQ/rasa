import datetime
import logging
import os
from typing import Any, Dict, List, Text, Type

from datadog_api_client.v1 import ApiClient, Configuration
from datadog_api_client.v1.api.metrics_api import MetricsApi
from datadog_api_client.v1.model.metrics_payload import MetricsPayload
from datadog_api_client.v1.model.point import Point
from datadog_api_client.v1.model.series import Series
from rasa.engine.caching import TrainingCache
from rasa.engine.graph import ExecutionContext, GraphNodeHook, GraphSchema, SchemaNode
from rasa.engine.storage.storage import ModelStorage
from rasa.engine.training.components import PrecomputedValueProvider
import rasa.shared.utils.io
from rasa.engine.training import fingerprinting

logger = logging.getLogger(__name__)
DD_ENV = "rasa-regression-tests"
DD_SERVICE = "rasa"
METRIC_COMP_PREFIX = "rasa.graph.comp."
CONFIG_REPOSITORY = "training-data"

MAIN_TAGS = {
    "config": "CONFIG",
    "dataset": "DATASET_NAME",
}

OTHER_TAGS = {
    "config_repository_branch": "DATASET_REPOSITORY_BRANCH",
    "dataset_commit": "DATASET_COMMIT",
    "accelerator_type": "ACCELERATOR_TYPE",
    "type": "TYPE",
    "index_repetition": "INDEX_REPETITION",
    "host_name": "HOST_NAME",
}

GIT_RELATED_TAGS = {
    "pr_id": "PR_ID",
    "pr_url": "PR_URL",
    "github_event": "GITHUB_EVENT_NAME",
    "github_run_id": "GITHUB_RUN_ID",
    "github_sha": "GITHUB_SHA",
    "workflow": "GITHUB_WORKFLOW",
}


def create_dict_of_env(name_to_env: Dict[Text, Text]) -> Dict[Text, Text]:
    return {name: os.environ[env_var] for name, env_var in name_to_env.items()}


def prepare_dsrepo_and_external_tags_as_str() -> Dict[Text, Text]:
    return {
        "dataset_repository_branch": os.environ["DATASET_REPOSITORY_BRANCH"],
        "external_dataset_repository": os.environ["IS_EXTERNAL"],
    }


def prepare_datadog_tags() -> List[Text]:
    tags = {
        "env": DD_ENV,
        "service": DD_SERVICE,
        "branch": os.environ["BRANCH"],
        "config_repository": CONFIG_REPOSITORY,
        **prepare_dsrepo_and_external_tags_as_str(),
        **create_dict_of_env(MAIN_TAGS),
        **create_dict_of_env(OTHER_TAGS),
        **create_dict_of_env(GIT_RELATED_TAGS),
    }
    tags_list = [f"{k}:{v}" for k, v in tags.items()]
    return tags_list


class TrainingHook(GraphNodeHook):
    """Caches fingerprints and outputs of nodes during model training."""

    def __init__(
        self,
        cache: TrainingCache,
        model_storage: ModelStorage,
        pruned_schema: GraphSchema,
    ) -> None:
        """Initializes a `TrainingHook`.

        Args:
            cache: Cache used to store fingerprints and outputs.
            model_storage: Used to cache `Resource`s.
            pruned_schema: The pruned training schema.
        """
        self._cache = cache
        self._model_storage = model_storage
        self._pruned_schema = pruned_schema

    def on_before_node(
        self,
        node_name: Text,
        execution_context: ExecutionContext,
        config: Dict[Text, Any],
        received_inputs: Dict[Text, Any],
    ) -> Dict:
        """Calculates the run fingerprint for use in `on_after_node`."""
        graph_component_class = self._get_graph_component_class(
            execution_context, node_name
        )

        fingerprint_key = fingerprinting.calculate_fingerprint_key(
            graph_component_class=graph_component_class,
            config=config,
            inputs=received_inputs,
        )

        return {"fingerprint_key": fingerprint_key}

    def on_after_node(
        self,
        node_name: Text,
        execution_context: ExecutionContext,
        config: Dict[Text, Any],
        output: Any,
        input_hook_data: Dict,
    ) -> None:
        """Stores the fingerprints and caches the output of the node."""
        # We should not re-cache the output of a PrecomputedValueProvider.
        graph_component_class = self._pruned_schema.nodes[node_name].uses

        if graph_component_class == PrecomputedValueProvider:
            return None

        output_fingerprint = rasa.shared.utils.io.deep_container_fingerprint(output)
        fingerprint_key = input_hook_data["fingerprint_key"]

        logger.debug(
            f"Caching '{output.__class__.__name__}' with fingerprint_key: "
            f"'{fingerprint_key}' and output_fingerprint '{output_fingerprint}'."
        )

        self._cache.cache_output(
            fingerprint_key=fingerprint_key,
            output=output,
            output_fingerprint=output_fingerprint,
            model_storage=self._model_storage,
        )

    @staticmethod
    def _get_graph_component_class(
        execution_context: ExecutionContext, node_name: Text
    ) -> Type:
        graph_component_class = execution_context.graph_schema.nodes[node_name].uses
        return graph_component_class


class LoggingHook(GraphNodeHook):
    """Logs the training of components."""

    def __init__(self, pruned_schema: GraphSchema) -> None:
        """Creates hook.

        Args:
            pruned_schema: The pruned schema provides us with the information whether
                a component is cached or not.
        """
        self._pruned_schema = pruned_schema

    def on_before_node(
        self,
        node_name: Text,
        execution_context: ExecutionContext,
        config: Dict[Text, Any],
        received_inputs: Dict[Text, Any],
    ) -> Dict:
        """Logs the training start of a graph node."""
        node = self._pruned_schema.nodes[node_name]

        if not self._is_cached_node(node) and self._does_node_train(node):
            logger.info(f"Starting to train component '{node.uses.__name__}'.")

        return {}

    @staticmethod
    def _does_node_train(node: SchemaNode) -> bool:
        # Nodes which train are always targets so that they store their output in the
        # model storage. `is_input` filters out nodes which don't really train but e.g.
        # persist some training data.
        return node.is_target and not node.is_input

    @staticmethod
    def _is_cached_node(node: SchemaNode) -> bool:
        return node.uses == PrecomputedValueProvider

    def on_after_node(
        self,
        node_name: Text,
        execution_context: ExecutionContext,
        config: Dict[Text, Any],
        output: Any,
        input_hook_data: Dict,
    ) -> None:
        """Logs when a component finished its training."""
        node = self._pruned_schema.nodes[node_name]

        if not self._does_node_train(node):
            return

        if self._is_cached_node(node):
            actual_component = execution_context.graph_schema.nodes[node_name]
            logger.info(
                f"Restored component '{actual_component.uses.__name__}' from cache."
            )
        else:
            logger.info(f"Finished training component '{node.uses.__name__}'.")


class DatadogHook(GraphNodeHook):
    """Sends training node information to Datadog."""

    def __init__(self, pruned_schema: GraphSchema) -> None:
        """Creates hook.

        Args:
            pruned_schema: The pruned schema provides us with the information whether
                a component is cached or not.
        """
        self._pruned_schema = pruned_schema

    def on_before_node(
        self,
        node_name: Text,
        execution_context: ExecutionContext,
        config: Dict[Text, Any],
        received_inputs: Dict[Text, Any],
    ) -> Dict:
        """Sends when a component starts its training."""
        node = self._pruned_schema.nodes[node_name]

        if not self._is_cached_node(node) and self._does_node_train(node):
            logger.info(f"Starting to train component '{node.uses.__name__}'.")

        node_id: int = self.get_node_id(node_name)
        self.send_to_datadog(node_id, node_name)

        return {}

    @staticmethod
    def send_to_datadog(node_id: int, node_name: str) -> None:
        """Sends metrics to datadog."""
        # Prepare
        tags_list = prepare_datadog_tags()
        if node_name:
            tags_list.append(f"node_name:{node_name}")
        timestamp = datetime.datetime.now().timestamp()
        series = []

        metric_name = "node_id"
        metric_value = node_id

        series.append(
            Series(
                metric=f"{METRIC_COMP_PREFIX}{metric_name}.gauge",
                type="gauge",
                points=[Point([timestamp, float(metric_value)])],
                tags=tags_list,
            )
        )

        body = MetricsPayload(series=series)
        with ApiClient(Configuration()) as api_client:
            api_instance = MetricsApi(api_client)
            response = api_instance.submit_metrics(body=body)
            if response.get("status") != "ok":
                print(response)

    @staticmethod
    def get_node_id(node_name: Text) -> int:
        """Extracts node ID from node name."""
        digit_list = ""
        for s in node_name[::-1]:
            if s.isdigit():
                digit_list += s
        if digit_list:
            node_id = int(digit_list[::-1])
        else:
            node_id = -2
        return node_id

    @staticmethod
    def _does_node_train(node: SchemaNode) -> bool:
        # Nodes which train are always targets so that they store their output in the
        # model storage. `is_input` filters out nodes which don't really train but e.g.
        # persist some training data.
        return node.is_target and not node.is_input

    @staticmethod
    def _is_cached_node(node: SchemaNode) -> bool:
        return node.uses == PrecomputedValueProvider

    def on_after_node(
        self,
        node_name: Text,
        execution_context: ExecutionContext,
        config: Dict[Text, Any],
        output: Any,
        input_hook_data: Dict,
    ) -> None:
        """Sends when a component finished its training."""
        node = self._pruned_schema.nodes[node_name]

        if not self._does_node_train(node):
            return

        node_id: int = -1
        self.send_to_datadog(node_id, node_name)

        if self._is_cached_node(node):
            actual_component = execution_context.graph_schema.nodes[node_name]
            logger.info(
                f"Restored component '{actual_component.uses.__name__}' from cache."
            )
        else:
            logger.info(f"Finished training component '{node.uses.__name__}'.")
