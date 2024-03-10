from __future__ import annotations
import typing
from pathlib import Path
from typing import Any, Dict, List, Optional, Text

from rasa.engine.graph import GraphComponent, ExecutionContext
from rasa.engine.recipes.default_recipe import DefaultV1Recipe
from rasa.engine.storage.resource import Resource
from rasa.engine.storage.storage import ModelStorage
from rasa.shared.exceptions import InvalidConfigException

if typing.TYPE_CHECKING:
    import mitie


class MitieModel:
    """Wraps `MitieNLP` output to make it fingerprintable."""

    def __init__(
        self,
        model_path: Path,
        word_feature_extractor: Optional["mitie.total_word_feature_extractor"] = None,
    ) -> None:
        """Initializing MitieModel."""
        import mitie

        self.word_feature_extractor = (
            word_feature_extractor or mitie.total_word_feature_extractor
        )
        self.model_path = model_path

    def fingerprint(self) -> Text:
        """Fingerprints the model path.

        Use a static fingerprint as we assume this only changes if the file path
        changes and want to avoid investigating the model in greater detail for now.

        Returns:
            Fingerprint for model.
        """
        return str(self.model_path)


@DefaultV1Recipe.register(
    DefaultV1Recipe.ComponentType.MODEL_LOADER, is_trainable=False
)
class MitieNLP(GraphComponent):
    """Component which provides the common configuration and loaded model to others.

    This is used to avoid loading the Mitie model multiple times. Instead the Mitie
    model is only loaded once and then shared by depending components.
    """

    @staticmethod
    def get_default_config() -> Dict[Text, Any]:
        """Returns default config (see parent class for full docstring)."""
        return {
            # name of the language model to load - this contains
            # the MITIE feature extractor
            "model": Path("data", "total_word_feature_extractor.dat")
        }

    def __init__(
        self,
        path_to_model_file: Path,
        extractor: Optional["mitie.total_word_feature_extractor"] = None,
    ) -> None:
        """Constructs a new language model from the MITIE framework."""
        self._path_to_model_file = path_to_model_file
        self._extractor = extractor

    @staticmethod
    def required_packages() -> List[Text]:
        """Lists required dependencies (see parent class for full docstring)."""
        return ["mitie"]

    @classmethod
    def create(
        cls,
        config: Dict[Text, Any],
        model_storage: ModelStorage,
        resource: Resource,
        execution_context: ExecutionContext,
    ) -> MitieNLP:
        """Creates component (see parent class for full docstring)."""
        import mitie

        model_file = config.get("model")
        if not model_file:
            raise InvalidConfigException(
                "The MITIE component 'MitieNLP' needs "
                "the configuration value for 'model'."
                "Please take a look at the "
                "documentation in the pipeline section "
                "to get more info about this "
                "parameter."
            )
        if not Path(model_file).is_file():
            raise InvalidConfigException(
                "The model file configured in the MITIE "
                "component cannot be found. "
                "Please ensure the directory path and/or "
                "filename, '{}', are correct.".format(model_file)
            )
        extractor = mitie.total_word_feature_extractor(str(model_file))

        return cls(Path(model_file), extractor)

    def provide(self) -> MitieModel:
        """Provides loaded `MitieModel` and path during training and inference."""
        return MitieModel(
            word_feature_extractor=self._extractor, model_path=self._path_to_model_file
        )
