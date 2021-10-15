import datetime
import logging
import os
from typing import Any, Dict, List, Optional, Text

import rasa.nlu
from rasa.shared.exceptions import RasaException
import rasa.shared.utils.io
import rasa.shared.utils.common
import rasa.utils.io
from rasa.nlu.config import component_config_from_pipeline
from rasa.nlu.utils import write_json_to_file

logger = logging.getLogger(__name__)


# TODO: remove/move
class InvalidModelError(RasaException):
    """Raised when a model failed to load.

    Attributes:
        message -- explanation of why the model is invalid
    """

    def __init__(self, message: Text) -> None:
        """Initialize message attribute."""
        self.message = message
        super(InvalidModelError, self).__init__(message)

    def __str__(self) -> Text:
        return self.message


class UnsupportedModelError(RasaException):
    """Raised when a model is too old to be loaded.

    Attributes:
        message -- explanation of why the model is invalid
    """

    def __init__(self, message: Text) -> None:
        """Initialize message attribute."""
        self.message = message
        super(UnsupportedModelError, self).__init__(message)

    def __str__(self) -> Text:
        return self.message


# TODO: remove once all components are removed
class Metadata:
    """Captures all information about a model to load and prepare it."""

    @staticmethod
    def load(model_dir: Text) -> "Metadata":
        """Loads the metadata from a models directory.

        Args:
            model_dir: the directory where the model is saved.
        Returns:
            Metadata: A metadata object describing the model
        """
        try:
            metadata_file = os.path.join(model_dir, "metadata.json")
            data = rasa.shared.utils.io.read_json_file(metadata_file)
            return Metadata(data)
        except Exception as e:
            abspath = os.path.abspath(os.path.join(model_dir, "metadata.json"))
            raise InvalidModelError(
                f"Failed to load model metadata from '{abspath}'. {e}"
            )

    def __init__(self, metadata: Dict[Text, Any]) -> None:
        """Set `metadata` attribute."""
        self.metadata = metadata

    def get(self, property_name: Text, default: Any = None) -> Any:
        """Proxy function to get property on `metadata` attribute."""
        return self.metadata.get(property_name, default)

    @property
    def component_classes(self) -> List[Optional[Text]]:
        """Returns a list of component class names."""
        if self.get("pipeline"):
            return [c.get("class") for c in self.get("pipeline", [])]
        else:
            return []

    @property
    def number_of_components(self) -> int:
        """Returns count of components."""
        return len(self.get("pipeline", []))

    def for_component(self, index: int, defaults: Any = None) -> Dict[Text, Any]:
        """Returns the configuration of the component based on index."""
        return component_config_from_pipeline(index, self.get("pipeline", []), defaults)

    @property
    def language(self) -> Optional[Text]:
        """Language of the underlying model"""

        return self.get("language")

    def persist(self, model_dir: Text) -> None:
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
