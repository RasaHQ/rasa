import datetime
import logging
import rasa.utils.io
import rasa.nlu

from typing import Any, Optional, Text, Dict
from rasa.constants import MINIMUM_COMPATIBLE_VERSION
from rasa.nlu.components.builder import ComponentBuilder
from rasa.nlu.components.pipeline import ComponentPipeline
from rasa.nlu.model.exceptions import UnsupportedModelError
from rasa.nlu.utils.package_manager import PackageManager
from rasa.nlu.model.metadata import Metadata
from rasa.nlu.training_data import Message
from rasa.nlu.components.exceptions import MissingArgumentError

logger = logging.getLogger(__name__)


class Interpreter:
    """Use a trained pipeline of components to parse text messages."""

    # Defines all attributes (& default values)
    # that will be returned by `parse`
    @staticmethod
    def default_output_attributes() -> Dict[Text, Any]:
        return {"intent": {"name": None, "confidence": 0.0}, "entities": []}

    @staticmethod
    def ensure_model_compatibility(
        metadata: Metadata, version_to_check: Optional[Text] = None
    ) -> None:
        from packaging import version

        if version_to_check is None:
            version_to_check = MINIMUM_COMPATIBLE_VERSION

        model_version = metadata.get("rasa_version", "0.0.0")
        if version.parse(model_version) < version.parse(version_to_check):
            raise UnsupportedModelError(
                "The model version is to old to be "
                "loaded by this Rasa NLU instance. "
                "Either retrain the model, or run with"
                "an older version. "
                "Model version: {} Instance version: {}"
                "".format(model_version, rasa.__version__)
            )

    @staticmethod
    def load(
        model_dir: Text,
        component_builder: Optional[ComponentBuilder] = None,
        skip_validation: bool = False,
    ) -> "Interpreter":
        """Create an interpreter based on a persisted model.

        Args:
            skip_validation: If set to `True`, tries to check that all
                required packages for the components are installed
                before loading them.
            model_dir: The path of the model to load
            component_builder: The
                :class:`rasa.nlu.components.ComponentBuilder` to use.

        Returns:
            An interpreter that uses the loaded model.
        """

        model_metadata = Metadata.load(model_dir)

        Interpreter.ensure_model_compatibility(model_metadata)
        return Interpreter.create(model_metadata, component_builder, skip_validation)

    @staticmethod
    def create(
        model_metadata: Metadata,
        component_builder: Optional[ComponentBuilder] = None,
        skip_validation: bool = False,
    ) -> "Interpreter":
        """Load stored model and components defined by the provided metadata."""

        context = {}

        if component_builder is None:
            # If no builder is passed, every interpreter creation will result
            # in a new builder. hence, no components are reused.
            component_builder = ComponentBuilder()

        pipeline = ComponentPipeline()

        # Before instantiating the component classes,
        # lets check if all required packages are available
        if not skip_validation:
            PackageManager.validate_requirements(model_metadata.component_classes)

        for i in range(model_metadata.number_of_components):
            component_meta = model_metadata.for_component(i)
            component = component_builder.load_component(
                component_meta, model_metadata.model_dir, model_metadata, **context
            )
            try:
                updates = component.provide_context()
                if updates:
                    context.update(updates)
                pipeline.add_component(component)
            except MissingArgumentError as e:
                raise Exception(
                    "Failed to initialize component '{}'. "
                    "{}".format(component.name, e)
                )

        return Interpreter(pipeline, context, model_metadata)

    def __init__(
        self,
        pipeline: ComponentPipeline,
        context: Optional[Dict[Text, Any]],
        model_metadata: Optional[Metadata] = None,
    ) -> None:

        self.pipeline = pipeline
        self.context = context if context is not None else {}
        self.model_metadata = model_metadata

    def parse(
        self,
        text: Text,
        time: Optional[datetime.datetime] = None,
        only_output_properties: bool = True,
    ) -> Dict[Text, Any]:
        """Parse the input text, classify it and return pipeline result.

        The pipeline result usually contains intent and entities."""

        if not text:
            # Not all components are able to handle empty strings. So we need
            # to prevent that... This default return will not contain all
            # output attributes of all components, but in the end, no one
            # should pass an empty string in the first place.
            output = self.default_output_attributes()
            output["text"] = ""
            return output

        message = Message(text, self.default_output_attributes(), time=time)

        for component in self.pipeline:
            component.process(message, **self.context)

        output = self.default_output_attributes()
        output.update(message.as_dict(only_output_properties=only_output_properties))
        return output
