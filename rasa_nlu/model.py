from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import datetime
import io
import json
import logging
import os
import copy

from builtins import object
from builtins import str
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Text

import rasa_nlu
from rasa_nlu import components
from rasa_nlu.components import Component
from rasa_nlu.components import ComponentBuilder
from rasa_nlu.config import RasaNLUConfig
from rasa_nlu.persistor import Persistor
from rasa_nlu.training_data import TrainingData


class InvalidModelError(Exception):
    """Raised when a model failed to load.

    Attributes:
        message -- explanation of why the model is invalid
    """

    def __init__(self, message):
        self.message = message

    def __str__(self):
        return self.message


class Metadata(object):
    """Captures all necessary information about a model to load it and prepare it for usage."""

    @staticmethod
    def load(model_dir):
        # type: (Text) -> 'Metadata'
        """Loads the metadata from a models directory."""
        try:
            with io.open(os.path.join(model_dir, 'metadata.json'), encoding="utf-8") as f:
                data = json.loads(f.read())
            return Metadata(data, model_dir)
        except Exception as e:
            raise InvalidModelError("Failed to load model metadata. {}".format(e))

    def __init__(self, metadata, model_dir):
        # type: (Dict[Text, Any], Optional[Text]) -> None

        self.metadata = metadata
        self.model_dir = model_dir

    def __prepend_path(self, prop):
        if self.metadata.get(prop) is not None:
            return os.path.normpath(os.path.join(self.model_dir, self.metadata[prop]))
        else:
            return None

    @property
    def language(self):
        # type: () -> Optional[Text]
        """Language of the underlying model"""

        return self.metadata.get('language')

    @property
    def pipeline(self):
        # type: () -> List[Text]
        """Names of the processing pipeline elements."""

        if 'pipeline' in self.metadata:
            return self.metadata['pipeline']
        elif 'backend' in self.metadata:   # This is for backwards compatibility of models trained before 0.8
            from rasa_nlu import registry
            return registry.registered_pipeline_templates.get(self.metadata.get('backend'))
        else:
            return []

    def persist(self, model_dir):
        # type: (Text) -> None
        """Persists the metadata of a model to a given directory."""

        metadata = self.metadata.copy()

        metadata.update({
            "trained_at": datetime.datetime.now().strftime('%Y%m%d-%H%M%S'),
            "rasa_nlu_version": rasa_nlu.__version__,
        })

        with io.open(os.path.join(model_dir, 'metadata.json'), 'w') as f:
            f.write(str(json.dumps(metadata, indent=4)))


class Trainer(object):
    """Given a pipeline specification and configuration this trainer will load the data and train all components."""

    # Officially supported languages (others might be used, but might fail)
    SUPPORTED_LANGUAGES = ["de", "en"]

    def __init__(self, config, component_builder=None, skip_validation=False):
        # type: (RasaNLUConfig, Optional[ComponentBuilder], bool) -> None

        self.config = config
        self.skip_validation = skip_validation
        self.training_data = None
        self.pipeline = []
        if component_builder is None:
            # If no builder is passed, every interpreter creation will result in a new builder.
            # hence, no components are reused.
            component_builder = components.ComponentBuilder()

        # Before instantiating the component classes, lets check if all required packages are available
        if not self.skip_validation:
            components.validate_requirements(config.pipeline)

        # Transform the passed names of the pipeline components into classes
        for component_name in config.pipeline:
            component = component_builder.create_component(component_name, config)
            self.pipeline.append(component)

    def train(self, data):
        # type: (TrainingData) -> Interpreter
        """Trains the underlying pipeline by using the provided training data."""

        # Before training the component classes, lets check if all arguments are provided
        if not self.skip_validation:
            components.validate_arguments(self.pipeline, self.config)

        self.training_data = data

        context = {}

        for component in self.pipeline:
            args = components.fill_args(component.pipeline_init_args(), context, self.config.as_dict())
            updates = component.pipeline_init(*args)
            if updates:
                context.update(updates)

        init_context = context.copy()

        context["training_data"] = data

        for component in self.pipeline:
            args = components.fill_args(component.train_args(), context, self.config.as_dict())
            logging.info("Starting to train component {}".format(component.name))
            updates = component.train(*args)
            logging.info("Finished training component.")
            if updates:
                context.update(updates)

        return Interpreter(self.pipeline, context=init_context, config=self.config.as_dict())

    def persist(self, path, persistor=None, model_name=None):
        # type: (Text, Optional[Persistor], bool) -> Text
        """Persist all components of the pipeline to the passed path. Returns the directory of the persited model."""

        timestamp = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
        metadata = {
            "language": self.config["language"],
            "pipeline": [component.name for component in self.pipeline],
        }

        if model_name is None:
            dir_name = os.path.join(path, "model_" + timestamp)
        else:
            dir_name = os.path.join(path, model_name)

        os.makedirs(dir_name)
        metadata.update(self.training_data.persist(dir_name))

        for component in self.pipeline:
            update = component.persist(dir_name)
            if update:
                metadata.update(update)

        Metadata(metadata, dir_name).persist(dir_name)

        if persistor is not None:
            persistor.save_tar(dir_name)
        logging.info("Successfully saved model into '{}'".format(os.path.abspath(dir_name)))
        return dir_name


class Interpreter(object):
    """Use a trained pipeline of components to parse text messages"""

    # Defines all attributes (and their default values) that will be returned by `parse`
    @staticmethod
    def default_output_attributes():
        return {"intent": {"name": "", "confidence": 0.0}, "entities": [], "text": ""}

    @staticmethod
    def load(meta, config, component_builder=None, skip_valdation=False):
        # type: (Metadata, RasaNLUConfig, Optional[ComponentBuilder], bool) -> Interpreter
        """Load a stored model and its components defined by the provided metadata."""
        context = Interpreter.default_output_attributes()
        context.update({"model_dir": meta.model_dir})

        if component_builder is None:
            # If no builder is passed, every interpreter creation will result in a new builder.
            # hence, no components are reused.
            component_builder = components.ComponentBuilder()

        model_config = config.as_dict()
        model_config.update(meta.metadata)

        pipeline = []

        # Before instantiating the component classes, lets check if all required packages are available
        if not skip_valdation:
            components.validate_requirements(meta.pipeline)

        for component_name in meta.pipeline:
            component = component_builder.load_component(component_name, context, model_config, meta)
            try:
                args = components.fill_args(component.pipeline_init_args(), context, model_config)
                updates = component.pipeline_init(*args)
                if updates:
                    context.update(updates)
                pipeline.append(component)
            except components.MissingArgumentError as e:
                raise Exception("Failed to initialize component '{}'. {}".format(component.name, e))

        return Interpreter(pipeline, context, model_config)

    def __init__(self, pipeline, context, config, meta=None):
        # type: (List[Component], Dict[Text, Any], Dict[Text, Any], Optional[Metadata]) -> None

        self.pipeline = pipeline
        self.context = context if context is not None else {}
        self.config = config
        self.meta = meta
        self.output_attributes = [output for component in pipeline for output in component.output_provides]

    def parse(self, text):
        # type: (Text) -> Dict[Text, Any]
        """Parse the input text, classify it and return an object containing its intent and entities."""

        if not text:
            # Not all components are able to handle empty strings. So we need to prevent that...
            # This default return will not contain all output attributes of all components,
            # but in the end, no one should pass an empty string in the first place.
            return self.default_output_attributes()

        current_context = self.context.copy()
        current_context.update(self.default_output_attributes())

        current_context.update({
            "text": text,
        })

        for component in self.pipeline:
            try:
                args = components.fill_args(component.process_args(), current_context, self.config)
                updates = component.process(*args)
                if updates:
                    current_context.update(updates)
            except components.MissingArgumentError as e:
                raise Exception("Failed to parse at component '{}'. {}".format(component.name, e))

        result = self.default_output_attributes()
        all_attributes = list(self.default_output_attributes().keys()) + self.output_attributes
        # Ensure only keys of `all_attributes` are present and no other keys are returned
        result.update({key: current_context[key] for key in all_attributes if key in current_context})
        return result
