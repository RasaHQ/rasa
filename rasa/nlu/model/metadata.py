import datetime
import logging
import os
import rasa.nlu
import rasa.utils.io

from typing import Any, Dict, Optional, Text
from rasa.nlu.config.manager import ConfigManager
from rasa.nlu.utils import write_json_to_file
from rasa.nlu.model.exceptions import InvalidModelError

logger = logging.getLogger(__name__)


class Metadata:
    """Captures all information about a model to load and prepare it."""

    @staticmethod
    def load(model_dir: Text):
        """Loads the metadata from a models directory.

        Args:
            model_dir: the directory where the model is saved.
        Returns:
            Metadata: A metadata object describing the model
        """
        try:
            metadata_file = os.path.join(model_dir, "metadata.json")
            data = rasa.utils.io.read_json_file(metadata_file)
            return Metadata(data, model_dir)
        except Exception as e:
            abspath = os.path.abspath(os.path.join(model_dir, "metadata.json"))
            raise InvalidModelError(
                "Failed to load model metadata from '{}'. {}".format(abspath, e)
            )

    def __init__(self, metadata: Dict[Text, Any], model_dir: Optional[Text]):

        self.metadata = metadata
        self.model_dir = model_dir

    def get(self, property_name, default=None):
        return self.metadata.get(property_name, default)

    @property
    def component_classes(self):
        if self.get("pipeline"):
            return [c.get("class") for c in self.get("pipeline", [])]
        else:
            return []

    @property
    def number_of_components(self):
        return len(self.get("pipeline", []))

    def for_component(self, index, defaults=None):
        return ConfigManager.component_config_from_pipeline(
            index, self.get("pipeline", []), defaults
        )

    @property
    def language(self) -> Optional[Text]:
        """Language of the underlying model"""

        return self.get("language")

    def persist(self, model_dir: Text):
        """Persists the metadata of a model to a given directory."""

        metadata = self.metadata.copy()

        metadata.update(
            {
                "trained_at": datetime.datetime.now().strftime("%Y%m%d-%H%M%S"),
                "rasa_version": rasa.__version__,
            }
        )

        filename = os.path.join(model_dir, "metadata.json")
        write_json_to_file(filename, metadata, indent=4)
