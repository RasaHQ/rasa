#Old file : nlu/components.py

import logging

from rasa.nlu.components.component import Component
from rasa.nlu.model.metadata import Metadata
from rasa.nlu.components.exceptions import MissingArgumentError
from rasa.nlu.config.nlu import RasaNLUModelConfig
from typing import Any, Dict, Optional, Text, Tuple

logger = logging.getLogger(__name__)


class ComponentBuilder:
    """Creates trainers and interpreters based on configurations.

    Caches components for reuse.
    """

    def __init__(self, use_cache: bool = True) -> None:
        self.use_cache = use_cache
        # Reuse nlp and featurizers where possible to save memory,
        # every component that implements a cache-key will be cached
        self.component_cache = {}

    def __get_cached_component(
        self, component_meta: Dict[Text, Any], model_metadata: Metadata
    ) -> Tuple[Optional[Component], Optional[Text]]:
        """Load a component from the cache, if it exists.

        Returns the component, if found, and the cache key.
        """

        from rasa.nlu.components import registry

        # try to get class name first, else create by name
        component_name = component_meta.get("class", component_meta["name"])
        component_class = registry.get_component_class(component_name)
        cache_key = component_class.cache_key(component_meta, model_metadata)
        if (
            cache_key is not None
            and self.use_cache
            and cache_key in self.component_cache
        ):
            return self.component_cache[cache_key], cache_key
        else:
            return None, cache_key

    def __add_to_cache(self, component: Component, cache_key: Optional[Text]) -> None:
        """Add a component to the cache."""

        if cache_key is not None and self.use_cache:
            self.component_cache[cache_key] = component
            logger.info(
                "Added '{}' to component cache. Key '{}'."
                "".format(component.name, cache_key)
            )

    def load_component(
        self,
        component_meta: Dict[Text, Any],
        model_dir: Text,
        model_metadata: "Metadata",
        **context: Any
    ) -> Component:
        """Tries to retrieve a component from the cache, else calls
        ``load`` to create a new component.

        Args:
            component_meta:
                the metadata of the component to load in the pipeline
            model_dir:
                the directory to read the model from
            model_metadata (Metadata):
                the model's :class:`rasa.nlu.model.Metadata`

        Returns:
            Component: the loaded component.
        """

        from rasa.nlu.components import registry

        try:
            cached_component, cache_key = self.__get_cached_component(
                component_meta, model_metadata
            )
            component = registry.load_component_by_meta(
                component_meta, model_dir, model_metadata, cached_component, **context
            )
            if not cached_component:
                # If the component wasn't in the cache,
                # let us add it if possible
                self.__add_to_cache(component, cache_key)
            return component
        except MissingArgumentError as e:  # pragma: no cover
            raise Exception(
                "Failed to load component from file `{}`. "
                "{}".format(component_meta.get("file"), e)
            )

    def create_component(
        self, component_config: Dict[Text, Any], cfg: RasaNLUModelConfig
    ) -> Component:
        """Tries to retrieve a component from the cache,
        calls `create` to create a new component."""
        from rasa.nlu.components import registry

        try:
            component, cache_key = self.__get_cached_component(
                component_config, Metadata(cfg.as_dict(), None)
            )
            if component is None:
                component = registry.create_component_by_config(component_config, cfg)
                self.__add_to_cache(component, cache_key)
            return component
        except MissingArgumentError as e:  # pragma: no cover
            raise Exception(
                "Failed to create component `{}`. "
                "{}".format(component_config["name"], e)
            )
