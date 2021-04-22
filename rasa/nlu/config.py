import copy
import logging
import os
from typing import Any, Dict, List, Optional, Text, Union

from rasa.shared.exceptions import InvalidConfigException
import rasa.shared.utils.io
import rasa.utils.io
from rasa.shared.constants import (
    DOCS_URL_PIPELINE,
    DEFAULT_CONFIG_PATH,
)
from rasa.shared.utils.io import json_to_string
from rasa.nlu.constants import COMPONENT_INDEX
import rasa.utils.train_utils

logger = logging.getLogger(__name__)


# DEPRECATED: will be removed in Rasa Open Source 3.0
InvalidConfigError = InvalidConfigException


def load(
    config: Optional[Union[Text, Dict]] = None, **kwargs: Any
) -> "RasaNLUModelConfig":
    """Create configuration from file or dict.

    Args:
        config: a file path, a dictionary with configuration keys. If set to
            `None` the configuration will be loaded from the default file
            path.

    Returns:
        Configuration object.
    """
    if isinstance(config, Dict):
        return _load_from_dict(config, **kwargs)

    file_config = {}
    if config is None and os.path.isfile(DEFAULT_CONFIG_PATH):
        config = DEFAULT_CONFIG_PATH

    if config is not None:
        file_config = rasa.shared.utils.io.read_model_configuration(config)

    return _load_from_dict(file_config, **kwargs)


def _load_from_dict(config: Dict, **kwargs: Any) -> "RasaNLUModelConfig":
    if kwargs:
        config.update(kwargs)
    return RasaNLUModelConfig(config)


def component_config_from_pipeline(
    index: int,
    pipeline: List[Dict[Text, Any]],
    defaults: Optional[Dict[Text, Any]] = None,
) -> Dict[Text, Any]:
    """Gets the configuration of the `index`th component.

    Args:
        index: Index of the component in the pipeline.
        pipeline: Configurations of the components in the pipeline.
        defaults: Default configuration.

    Returns:
        The `index`th component configuration, expanded
        by the given defaults.
    """
    try:
        configuration = copy.deepcopy(pipeline[index])
        configuration[COMPONENT_INDEX] = index
        return rasa.utils.train_utils.override_defaults(defaults, configuration)
    except IndexError:
        rasa.shared.utils.io.raise_warning(
            f"Tried to get configuration value for component "
            f"number {index} which is not part of your pipeline. "
            f"Returning `defaults`.",
            docs=DOCS_URL_PIPELINE,
        )
        return rasa.utils.train_utils.override_defaults(
            defaults, {COMPONENT_INDEX: index}
        )


class RasaNLUModelConfig:
    """A class that stores NLU model configuration parameters."""

    def __init__(self, configuration_values: Optional[Dict[Text, Any]] = None) -> None:
        """Create a model configuration.

        Args:
            configuration_values: optional dictionary to override defaults.
        """
        if not configuration_values:
            configuration_values = {}

        self.language = "en"
        self.pipeline = []
        self.data = None

        self.override(configuration_values)

        if self.__dict__["pipeline"] is None:
            # replaces None with empty list
            self.__dict__["pipeline"] = []

        for key, value in self.items():
            setattr(self, key, value)

    def __getitem__(self, key: Text) -> Any:
        return self.__dict__[key]

    def get(self, key: Text, default: Any = None) -> Any:
        return self.__dict__.get(key, default)

    def __setitem__(self, key: Text, value: Any) -> None:
        self.__dict__[key] = value

    def __delitem__(self, key: Text) -> None:
        del self.__dict__[key]

    def __contains__(self, key: Text) -> bool:
        return key in self.__dict__

    def __len__(self) -> int:
        return len(self.__dict__)

    def __getstate__(self) -> Dict[Text, Any]:
        return self.as_dict()

    def __setstate__(self, state: Dict[Text, Any]) -> None:
        self.override(state)

    def items(self) -> List[Any]:
        return list(self.__dict__.items())

    def as_dict(self) -> Dict[Text, Any]:
        return dict(list(self.items()))

    def view(self) -> Text:
        return json_to_string(self.__dict__, indent=4)

    def for_component(
        self, index: int, defaults: Optional[Dict[Text, Any]] = None
    ) -> Dict[Text, Any]:
        return component_config_from_pipeline(index, self.pipeline, defaults)

    @property
    def component_names(self) -> List[Text]:
        if self.pipeline:
            return [c.get("name") for c in self.pipeline]
        else:
            return []

    def set_component_attr(self, index: int, **kwargs: Any) -> None:
        try:
            self.pipeline[index].update(kwargs)
        except IndexError:
            rasa.shared.utils.io.raise_warning(
                f"Tried to set configuration value for component "
                f"number {index} which is not part of the pipeline.",
                docs=DOCS_URL_PIPELINE,
            )

    def override(self, config: Optional[Dict[Text, Any]] = None) -> None:
        """Overrides default config with given values.

        Args:
            config: New values for the configuration.
        """
        if config:
            self.__dict__.update(config)
