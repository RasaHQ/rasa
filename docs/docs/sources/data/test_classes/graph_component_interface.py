from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List, Type, Dict, Text, Any, Optional

from rasa.engine.graph import ExecutionContext
from rasa.engine.storage.resource import Resource
from rasa.engine.storage.storage import ModelStorage


class GraphComponent(ABC):
    """Interface for any component which will run in a graph."""

    @classmethod
    def required_components(cls) -> List[Type]:
        """Components that should be included in the pipeline before this component."""
        return []

    @classmethod
    @abstractmethod
    def create(
        cls,
        config: Dict[Text, Any],
        model_storage: ModelStorage,
        resource: Resource,
        execution_context: ExecutionContext,
    ) -> GraphComponent:
        """Creates a new `GraphComponent`.

        Args:
            config: This config overrides the `default_config`.
            model_storage: Storage which graph components can use to persist and load
                themselves.
            resource: Resource locator for this component which can be used to persist
                and load itself from the `model_storage`.
            execution_context: Information about the current graph run.

        Returns: An instantiated `GraphComponent`.
        """
        ...

    @classmethod
    def load(
        cls,
        config: Dict[Text, Any],
        model_storage: ModelStorage,
        resource: Resource,
        execution_context: ExecutionContext,
        **kwargs: Any,
    ) -> GraphComponent:
        """Creates a component using a persisted version of itself.

        If not overridden this method merely calls `create`.

        Args:
            config: The config for this graph component. This is the default config of
                the component merged with config specified by the user.
            model_storage: Storage which graph components can use to persist and load
                themselves.
            resource: Resource locator for this component which can be used to persist
                and load itself from the `model_storage`.
            execution_context: Information about the current graph run.
            kwargs: Output values from previous nodes might be passed in as `kwargs`.

        Returns:
            An instantiated, loaded `GraphComponent`.
        """
        return cls.create(config, model_storage, resource, execution_context)

    @staticmethod
    def get_default_config() -> Dict[Text, Any]:
        """Returns the component's default config.

        Default config and user config are merged by the `GraphNode` before the
        config is passed to the `create` and `load` method of the component.

        Returns:
            The default config of the component.
        """
        return {}

    @staticmethod
    def supported_languages() -> Optional[List[Text]]:
        """Determines which languages this component can work with.

        Returns: A list of supported languages, or `None` to signify all are supported.
        """
        return None

    @staticmethod
    def not_supported_languages() -> Optional[List[Text]]:
        """Determines which languages this component cannot work with.

        Returns: A list of not supported languages, or
            `None` to signify all are supported.
        """
        return None

    @staticmethod
    def required_packages() -> List[Text]:
        """Any extra python dependencies required for this component to run."""
        return []

    @classmethod
    def fingerprint_addon(cls, config: Dict[str, Any]) -> Optional[str]:
        """Adds additional data to the fingerprint calculation.

        This is useful if a component uses external data that is not provided
        by the graph.
        """
        return None
