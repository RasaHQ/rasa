import copy
import logging
import os
import ruamel.yaml as yaml
import rasa.utils.io

from typing import Any, Dict, List, Optional, Text, Union
from rasa.nlu.config.exceptions import InvalidConfigError
from rasa.nlu.config.nlu import RasaNLUModelConfig
from rasa.constants import DEFAULT_CONFIG_PATH

logger = logging.getLogger(__name__)


class ConfigManager:
    @staticmethod
    def override_defaults(
        defaults: Optional[Dict[Text, Any]], custom: Optional[Dict[Text, Any]]
    ) -> Dict[Text, Any]:
        if defaults:
            cfg = copy.deepcopy(defaults)
        else:
            cfg = {}

        if custom:
            cfg.update(custom)
        return cfg

    @staticmethod
    def component_config_from_pipeline(
        index: int,
        pipeline: List[Dict[Text, Any]],
        defaults: Optional[Dict[Text, Any]] = None,
    ) -> Dict[Text, Any]:
        try:
            c = pipeline[index]
            return ConfigManager.override_defaults(defaults, c)
        except IndexError:
            logger.warning(
                "Tried to get configuration value for component "
                "number {} which is not part of the pipeline. "
                "Returning `defaults`."
                "".format(index)
            )
            return ConfigManager.override_defaults(defaults, {})

    @staticmethod
    def load(
        config: Optional[Union[Text, Dict]] = None, **kwargs: Any
    ) -> RasaNLUModelConfig:
        if isinstance(config, Dict):
            return ConfigManager.__load_from_dict(config, **kwargs)

        file_config = {}
        if config is None and os.path.isfile(DEFAULT_CONFIG_PATH):
            config = DEFAULT_CONFIG_PATH

        if config is not None:
            try:
                file_config = rasa.utils.io.read_config_file(config)
            except yaml.parser.ParserError as e:
                raise InvalidConfigError(
                    "Failed to read configuration file '{}'. Error: {}".format(
                        config, e
                    )
                )

        return ConfigManager.__load_from_dict(file_config, **kwargs)

    @staticmethod
    def __load_from_dict(config: Dict, **kwargs: Any) -> RasaNLUModelConfig:
        if kwargs:
            config.update(kwargs)
        return RasaNLUModelConfig(config)
