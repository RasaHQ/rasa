import copy
import datetime
import logging
import os
from contextlib import ExitStack
from pathlib import Path
from typing import Any, Dict, List, Optional, Text, Tuple

import rasa.nlu
from rasa.nlu.featurizers.featurizer import Featurizer
from rasa.nlu.tokenizers.tokenizer import Tokenizer
from rasa.shared.exceptions import RasaException
import rasa.shared.utils.io
import rasa.utils.io
from rasa.constants import MINIMUM_COMPATIBLE_VERSION, NLU_MODEL_NAME_PREFIX
from rasa.nlu import components, utils
from rasa.nlu.classifiers.classifier import IntentClassifier
from rasa.nlu.components import Component, ComponentBuilder
from rasa.nlu.config import RasaNLUModelConfig, component_config_from_pipeline
from rasa.nlu.extractors.extractor import EntityExtractor

from rasa.nlu.persistor import Persistor
from rasa.shared.nlu.constants import (
    TEXT,
    ENTITIES,
    INTENT,
    INTENT_NAME_KEY,
    PREDICTED_CONFIDENCE_KEY,
)
from rasa.shared.nlu.training_data.training_data import (
    NLUTrainingDataFull,
    NLUTrainingDataChunk,
)
from rasa.shared.nlu.training_data.message import Message
from rasa.nlu.utils import write_json_to_file
from rasa.utils.common import TempDirectoryPath
from rasa.shared.core.domain import Domain

logger = logging.getLogger(__name__)


class InvalidModelError(RasaException):
    """Raised when a model failed to load.

    Attributes:
        message -- explanation of why the model is invalid
    """

    def __init__(self, message: Text) -> None:
        self.message = message
        super(InvalidModelError, self).__init__()

    def __str__(self) -> Text:
        return self.message


class UnsupportedModelError(RasaException):
    """Raised when a model is too old to be loaded.

    Attributes:
        message -- explanation of why the model is invalid
    """

    def __init__(self, message: Text) -> None:
        self.message = message
        super(UnsupportedModelError, self).__init__()

    def __str__(self) -> Text:
        return self.message


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
            data = rasa.shared.utils.io.read_json_file(metadata_file)
            return Metadata(data, model_dir)
        except Exception as e:
            abspath = os.path.abspath(os.path.join(model_dir, "metadata.json"))
            raise InvalidModelError(
                f"Failed to load model metadata from '{abspath}'. {e}"
            )

    def __init__(self, metadata: Dict[Text, Any], model_dir: Optional[Text]):

        self.metadata = metadata
        self.model_dir = model_dir

    def get(self, property_name: Text, default: Any = None) -> Any:
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

    def for_component(self, index: int, defaults: Any = None) -> Dict[Text, Any]:
        return component_config_from_pipeline(index, self.get("pipeline", []), defaults)

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


class Trainer:
    """Trainer will load the data and train all components.

    Requires a pipeline specification and configuration to use for
    the training.
    """

    def __init__(
        self,
        config: RasaNLUModelConfig,
        component_builder: Optional[ComponentBuilder] = None,
        skip_validation: bool = False,
        domain: Optional[Domain] = None,
    ):
        """Intializes the trainer for loading data and training components."""
        self.config = config
        self.skip_validation = skip_validation
        self.training_data = None  # type: Optional[NLUTrainingDataFull]

        if component_builder is None:
            # If no builder is passed, every interpreter creation will result in
            # a new builder. hence, no components are reused.
            component_builder = components.ComponentBuilder()

        # Before instantiating the component classes, lets check if all
        # required packages are available
        if not self.skip_validation:
            components.validate_requirements(config.component_names)

        # build pipeline
        self.pipeline = self._build_pipeline(config, component_builder, domain)

    def _build_pipeline(
        self,
        model_config: RasaNLUModelConfig,
        component_builder: ComponentBuilder,
        domain: Optional[Domain] = None,
    ) -> List[Component]:
        """Transform the passed names of the pipeline components into classes."""
        pipeline = []

        # Transform the passed names of the pipeline components into classes
        for index, pipeline_component in enumerate(model_config.pipeline):
            component_config = model_config.for_component(index)
            component = component_builder.create_component(
                component_config, model_config, domain
            )
            components.validate_component_keys(component, pipeline_component)
            pipeline.append(component)

        if not self.skip_validation:
            components.validate_pipeline(pipeline)

        return pipeline

    def train(self, data: NLUTrainingDataFull, **kwargs: Any) -> "Interpreter":
        """Trains the underlying pipeline using the provided training data."""

        self.training_data = data

        self.training_data.validate()

        context = kwargs

        for component in self.pipeline:
            updates = component.provide_context()
            if updates:
                context.update(updates)

        # Before the training starts: check that all arguments are provided
        if not self.skip_validation:
            components.validate_required_components_from_data(
                self.pipeline, self.training_data
            )

        # data gets modified internally during the training - hence the copy
        working_data: NLUTrainingDataFull = copy.deepcopy(data)

        for i, component in enumerate(self.pipeline):
            logger.info(f"Starting to train component {component.name}")
            component.prepare_partial_processing(self.pipeline[:i], context)
            updates = component.train(working_data, self.config, **context)
            logger.info("Finished training component.")
            if updates:
                context.update(updates)

        return Interpreter(self.pipeline, context)

    def train_in_chunks(
        self,
        training_data: NLUTrainingDataFull,
        train_path: Path,
        number_of_chunks: int,
        fixed_model_name: Optional[Text] = None,
    ) -> Tuple["Interpreter", Text]:
        """Trains the underlying pipeline in chunks using the provided training data.

        Args:
            training_data: The training data.
            train_path: The training path to use.
            fixed_model_name: The model name to use.
            number_of_chunks: The number of chunks to use.

        Returns:
            The trained interpreter and the path to the persisted model.
        """
        import tempfile

        # TODO do we really need this copy? can we avoid it somehow?
        working_training_data = copy.deepcopy(training_data)
        working_training_data.validate()

        context = {}
        for component in self.pipeline:
            updates = component.provide_context()
            if updates:
                context.update(updates)

        # Before the training starts: check that all arguments are provided
        if not self.skip_validation:
            components.validate_required_components_from_data(
                self.pipeline, working_training_data
            )

        metadata = {"language": self.config["language"], "pipeline": []}

        data_chunk_dir = TempDirectoryPath(tempfile.mkdtemp())

        dir_name, model_name = self._create_model_dir(str(train_path), fixed_model_name)

        # perform tokenization & prepare other components for training in chunks
        for i, component in enumerate(self.pipeline):
            if isinstance(component, Tokenizer):
                component.train(working_training_data, self.config, **context)
                metadata["pipeline"].append(
                    self._persist_component(component, dir_name, i)
                )
            else:
                component.prepare_partial_training(working_training_data)

        training_data_chunks = working_training_data.divide_into_chunks(
            number_of_chunks
        )

        # perform featurization
        for i, data_chunk in enumerate(training_data_chunks):
            for component in self.pipeline:
                if isinstance(component, Featurizer):
                    component.train_chunk(data_chunk, self.config, **context)
            data_chunk.persist_chunk(data_chunk_dir, f"{i}_chunk.tfrecord")

        # persist featurizers
        for i, component in enumerate(self.pipeline):
            if isinstance(component, Featurizer):
                metadata["pipeline"].append(
                    self._persist_component(component, dir_name, i)
                )

        # TODO training of classifiers probably needs to be adapted
        for i, component in enumerate(self.pipeline):
            if isinstance(component, (IntentClassifier, EntityExtractor)):
                for j in range(number_of_chunks):
                    file_path = os.path.join(data_chunk_dir, f"{j}_chunk.tfrecord")
                    data_chunk = NLUTrainingDataChunk.load_chunk(file_path)
                    component.train_chunk(data_chunk, self.config, **context)
                metadata["pipeline"].append(
                    self._persist_component(component, dir_name, i)
                )

        Metadata(metadata, dir_name).persist(dir_name)

        logger.info(
            "Successfully saved model into '{}'.".format(os.path.abspath(dir_name))
        )

        return Interpreter(self.pipeline, context), dir_name

    @staticmethod
    def _create_model_dir(
        path: Text, fixed_model_name: Optional[Text] = None
    ) -> Tuple[Text, Text]:
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

        if fixed_model_name:
            model_name = fixed_model_name
        else:
            model_name = NLU_MODEL_NAME_PREFIX + timestamp

        path = os.path.abspath(path)
        dir_name = os.path.join(path, model_name)
        rasa.shared.utils.io.create_directory(str(dir_name))

        return dir_name, model_name

    @staticmethod
    def _file_name(index: int, name: Text) -> Text:
        return f"component_{index}_{name}"

    def persist(
        self,
        path: Text,
        persistor: Optional[Persistor] = None,
        fixed_model_name: Text = None,
        persist_nlu_training_data: bool = False,
    ) -> Text:
        """Persists all components of the pipeline to the passed path.

        Returns the directory of the persisted model.

        Args:
            path: The path to use.
            persistor: The persistor to use.
            fixed_model_name: The model name to use.
            persist_nlu_training_data: If True the training data will be persisted.

        Returns:
            The path to the directory the model is stored.
        """
        metadata = {"language": self.config["language"], "pipeline": []}

        dir_name, model_name = self._create_model_dir(path, fixed_model_name)

        if self.training_data and persist_nlu_training_data:
            metadata.update(self.training_data.persist(dir_name))

        for i, component in enumerate(self.pipeline):
            metadata["pipeline"].append(self._persist_component(component, dir_name, i))

        Metadata(metadata, dir_name).persist(dir_name)

        if persistor is not None:
            persistor.persist(dir_name, model_name)
        logger.info(
            "Successfully saved model into '{}'".format(os.path.abspath(dir_name))
        )
        return dir_name

    def _persist_component(
        self, component: Component, dir_name: Text, component_number: int
    ) -> Dict[Text, Any]:
        """Persists one component.

        Args:
            component: the component to persist
            dir_name: the directory to store the component to
            component_number: the component number (index in the pipeline)

        Returns:
            Metadata about the component and its configuration.
        """
        file_name = self._file_name(component_number, component.name)
        update = component.persist(file_name, dir_name)
        component_meta = component.component_config
        if update:
            component_meta.update(update)
        component_meta["class"] = utils.module_path_from_object(component)
        return component_meta


class Interpreter:
    """Use a trained pipeline of components to parse text messages."""

    # Defines all attributes (& default values)
    # that will be returned by `parse`
    @staticmethod
    def default_output_attributes() -> Dict[Text, Any]:
        return {
            TEXT: "",
            INTENT: {INTENT_NAME_KEY: None, PREDICTED_CONFIDENCE_KEY: 0.0},
            ENTITIES: [],
        }

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
                "The model version is too old to be "
                "loaded by this Rasa NLU instance. "
                "Either retrain the model, or run with "
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
            skip_validation: If set to `True`, does not check that all
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
            component_builder = components.ComponentBuilder()

        pipeline = []

        # Before instantiating the component classes,
        # lets check if all required packages are available
        if not skip_validation:
            components.validate_requirements(model_metadata.component_classes)

        for i in range(model_metadata.number_of_components):
            component_meta = model_metadata.for_component(i)
            component = component_builder.load_component(
                component_meta, model_metadata.model_dir, model_metadata, **context
            )
            try:
                updates = component.provide_context()
                if updates:
                    context.update(updates)
                pipeline.append(component)
            except components.MissingArgumentError as e:
                raise Exception(
                    "Failed to initialize component '{}'. "
                    "{}".format(component.name, e)
                )

        return Interpreter(pipeline, context, model_metadata)

    def __init__(
        self,
        pipeline: List[Component],
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

        data = self.default_output_attributes()
        data[TEXT] = text

        message = Message(data=data, time=time)

        for component in self.pipeline:
            component.process(message, **self.context)

        output = self.default_output_attributes()
        output.update(message.as_dict(only_output_properties=only_output_properties))
        return output

    def featurize_message(self, message: Message) -> Message:
        """
        Tokenize and featurize the input message
        Args:
            message: message storing text to process;
        Returns:
            message: it contains the tokens and features which are the output of the NLU pipeline;
        """

        for component in self.pipeline:
            if not isinstance(component, (EntityExtractor, IntentClassifier)):
                component.process(message, **self.context)
        return message
