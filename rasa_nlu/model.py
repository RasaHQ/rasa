from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import copy
import datetime
import logging
import os

from builtins import object
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Text

import rasa_nlu
from rasa_nlu import components, utils
from rasa_nlu.components import Component
from rasa_nlu.components import ComponentBuilder
from rasa_nlu.config import RasaNLUConfig
from rasa_nlu.persistor import Persistor
from rasa_nlu.training_data import TrainingData, Message
from rasa_nlu.utils import create_dir, write_json_to_file

logger = logging.getLogger(__name__)


class InvalidProjectError(Exception):
    """Raised when a model failed to load.

    Attributes:
        message -- explanation of why the model is invalid
    """

    def __init__(self, message):
        self.message = message

    def __str__(self):
        return self.message


class Metadata(object):
    """Captures all information about a model to load and prepare it."""

    @staticmethod
    def load(model_dir):
        # type: (Text) -> 'Metadata'
        """Loads the metadata from a models directory."""
        try:
            metadata_file = os.path.join(model_dir, 'metadata.json')
            data = utils.read_json_file(metadata_file)
            return Metadata(data, model_dir)
        except Exception as e:
            abspath = os.path.abspath(os.path.join(model_dir, 'metadata.json'))
            raise InvalidProjectError("Failed to load model metadata "
                                      "from '{}'. {}".format(abspath, e))

    def __init__(self, metadata, model_dir):
        # type: (Dict[Text, Any], Optional[Text]) -> None

        self.metadata = metadata
        self.model_dir = model_dir

    def get(self, property_name, default=None):
        return self.metadata.get(property_name, default)

    @property
    def language(self):
        # type: () -> Optional[Text]
        """Language of the underlying model"""

        return self.get('language')

    @property
    def pipeline(self):
        # type: () -> List[Text]
        """Names of the processing pipeline elements."""

        return self.get('pipeline', [])

    def persist(self, model_dir):
        # type: (Text) -> None
        """Persists the metadata of a model to a given directory."""

        metadata = self.metadata.copy()

        metadata.update({
            "trained_at": datetime.datetime.now().strftime('%Y%m%d-%H%M%S'),
            "rasa_nlu_version": rasa_nlu.__version__,
        })

        filename = os.path.join(model_dir, 'metadata.json')
        write_json_to_file(filename, metadata, indent=4)


class Trainer(object):
    """Trainer will load the data and train all components.

    Requires a pipeline specification and configuration to use for
    the training."""

    # Officially supported languages (others might be used, but might fail)
    SUPPORTED_LANGUAGES = ["de", "en"]

    def __init__(self, config, component_builder=None, skip_validation=False):
        # type: (RasaNLUConfig, Optional[ComponentBuilder], bool) -> None

        self.config = config
        self.skip_validation = skip_validation
        self.training_data = None  # type: Optional[TrainingData]
        self.pipeline = []  # type: List[Component]
        if component_builder is None:
            # If no builder is passed, every interpreter creation will result in
            # a new builder. hence, no components are reused.
            component_builder = components.ComponentBuilder()

        # Before instantiating the component classes, lets check if all
        # required packages are available
        if not self.skip_validation:
            components.validate_requirements(config.pipeline)

        # Transform the passed names of the pipeline components into classes
        for component_name in config.pipeline:
            component = component_builder.create_component(
                    component_name, config)
            self.pipeline.append(component)

    def train(self, data):
        # type: (TrainingData) -> Interpreter
        """Trains the underlying pipeline using the provided training data."""

        self.training_data = data

        context = {}  # type: Dict[Text, Any]

        for component in self.pipeline:
            updates = component.provide_context()
            if updates:
                context.update(updates)

        # Before the training starts: check that all arguments are provided
        if not self.skip_validation:
            components.validate_arguments(self.pipeline, context)

        # data gets modified internally during the training - hence the copy
        working_data = copy.deepcopy(data)

        for i, component in enumerate(self.pipeline):
            logger.info("Starting to train component {}".format(component.name))
            component.prepare_partial_processing(self.pipeline[:i], context)
            updates = component.train(working_data, self.config, **context)
            logger.info("Finished training component.")
            if updates:
                context.update(updates)

        return Interpreter(self.pipeline, context)

    def persist(self, path, persistor=None, project_name=None,
                fixed_model_name=None):
        # type: (Text, Optional[Persistor], Text) -> Text
        """Persist all components of the pipeline to the passed path.

        Returns the directory of the persisted model."""

        timestamp = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
        metadata = {
            "language": self.config["language"],
            "pipeline": [utils.module_path_from_object(component)
                         for component in self.pipeline],
        }

        if project_name is None:
            project_name = "default"

        if fixed_model_name:
            model_name = fixed_model_name
        else:
            model_name = "model_" + timestamp
        dir_name = os.path.join(path, project_name, model_name)

        create_dir(dir_name)

        if self.training_data:
            metadata.update(self.training_data.persist(dir_name))

        for component in self.pipeline:
            update = component.persist(dir_name)
            if update:
                metadata.update(update)

        Metadata(metadata, dir_name).persist(dir_name)

        if persistor is not None:
            persistor.persist(dir_name, model_name, project_name)
        logger.info("Successfully saved model into "
                    "'{}'".format(os.path.abspath(dir_name)))
        return dir_name


class Interpreter(object):
    """Use a trained pipeline of components to parse text messages"""

    # Defines all attributes (& default values) that will be returned by `parse`
    @staticmethod
    def default_output_attributes():
        return {"intent": {"name": "", "confidence": 0.0}, "entities": []}

    @staticmethod
    def load(model_dir, config=RasaNLUConfig(), component_builder=None,
             skip_valdation=False):
        """Creates an interpreter based on a persisted model."""

        if isinstance(model_dir, Metadata):
            # this is for backwards compatibilities (metadata passed as a dict)
            model_metadata = model_dir
            logger.warn("Deprecated use of `Interpreter.load` with a metadata "
                        "object. If you want to directly pass the metadata, "
                        "use `Interpreter.create(metadata, ...)`. If you want "
                        "to load the metadata from file, use "
                        "`Interpreter.load(model_dir, ...)")
        else:
            model_metadata = Metadata.load(model_dir)
        return Interpreter.create(model_metadata, config, component_builder,
                                  skip_valdation)

    @staticmethod
    def create(model_metadata,  # type: Metadata
               config,  # type: RasaNLUConfig
               component_builder=None,  # type: Optional[ComponentBuilder]
               skip_valdation=False  # type: bool
               ):
        # type: (...) -> Interpreter
        """Load stored model and components defined by the provided metadata."""

        context = {}

        if component_builder is None:
            # If no builder is passed, every interpreter creation will result
            # in a new builder. hence, no components are reused.
            component_builder = components.ComponentBuilder()

        pipeline = []

        # Before instantiating the component classes,
        # lets check if all required packages are available
        if not skip_valdation:
            components.validate_requirements(model_metadata.pipeline)

        for component_name in model_metadata.pipeline:
            component = component_builder.load_component(
                    component_name, model_metadata.model_dir,
                    model_metadata, config=config, **context)
            try:
                updates = component.provide_context()
                if updates:
                    context.update(updates)
                pipeline.append(component)
            except components.MissingArgumentError as e:
                raise Exception("Failed to initialize component '{}'. "
                                "{}".format(component.name, e))

        return Interpreter(pipeline, context, model_metadata)

    def __init__(self, pipeline, context, model_metadata=None):
        # type: (List[Component], Dict[Text, Any], Optional[Metadata]) -> None

        self.pipeline = pipeline
        self.context = context if context is not None else {}
        self.model_metadata = model_metadata

    def parse(self, text, time=None, only_output_properties=True):
        # type: (Text) -> Dict[Text, Any]
        """Parse the input text, classify it and return pipeline result.

        The pipeline result usually contains intent and entities."""

        if not text:
            # Not all components are able to handle empty strings. So we need
            # to prevent that... This default return will not contain all
            # output attributes of all components, but in the end, no one should
            # pass an empty string in the first place.
            output = self.default_output_attributes()
            output["text"] = ""
            return output

        message = Message(text, self.default_output_attributes(), time=time)

        for component in self.pipeline:
            component.process(message, **self.context)

        output = self.default_output_attributes()
        output.update(message.as_dict(only_output_properties=only_output_properties))
        return output
