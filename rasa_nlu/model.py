from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from builtins import str
from builtins import object
import datetime
import json
import logging
import os
import io

from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Text

import rasa_nlu.components
from rasa_nlu.components import Component
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


class Metadata(object):
    """Captures all necessary information about a model to load it and prepare it for usage."""

    @staticmethod
    def load(model_dir):
        # type: (Text) -> 'Metadata'
        """Loads the metadata from a models directory."""

        with io.open(os.path.join(model_dir, 'metadata.json'), encoding="utf-8") as f:
            data = json.loads(f.read())
        return Metadata(data, model_dir)

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

    def __init__(self, config, component_builder=None):
        # type: (RasaNLUConfig, Optional[rasa_nlu.components.ComponentBuilder]) -> None

        self.config = config
        self.training_data = None
        self.pipeline = []
        if component_builder is None:
            # If no builder is passed, every interpreter creation will result in a new builder.
            # hence, no components are reused.
            component_builder = rasa_nlu.components.ComponentBuilder()

        # Transform the passed names of the pipeline components into classes
        for component_name in config.pipeline:
            component = component_builder.create_component(component_name, config)
            self.pipeline.append(component)

    def validate(self, allow_empty_pipeline=False):
        # type: (bool) -> None
        """Validates a pipeline before it is run. Ensures, that all arguments are present to train the pipeline."""

        # Ensure the pipeline is not empty
        if not allow_empty_pipeline and len(self.pipeline) == 0:
            raise ValueError("Can not train an empty pipeline. " +
                             "Make sure to specify a proper pipeline in the configuration using the `pipeline` key." +
                             "The `backend` configuration key is NOT supported anymore.")

        # Validate the init phase
        context = {}

        for component in self.pipeline:
            try:
                rasa_nlu.components.fill_args(component.pipeline_init_args(), context, self.config.as_dict())
                updates = component.context_provides.get("pipeline_init", [])
                for u in updates:
                    context[u] = None
            except rasa_nlu.components.MissingArgumentError as e:
                raise Exception("Failed to validate at component '{}'. {}".format(component.name, e.message))

        after_init_context = context.copy()

        context["training_data"] = None     # Prepare context for testing the training phase

        for component in self.pipeline:
            try:
                rasa_nlu.components.fill_args(component.train_args(), context, self.config.as_dict())
                updates = component.context_provides.get("train", [])
                for u in updates:
                    context[u] = None
            except rasa_nlu.components.MissingArgumentError as e:
                raise Exception("Failed to validate at component '{}'. {}".format(component.name, e.message))

        # Reset context to test processing phase and prepare for training phase
        context = after_init_context
        context["text"] = None

        for component in self.pipeline:
            try:
                rasa_nlu.components.fill_args(component.process_args(), context, self.config.as_dict())
                updates = component.context_provides.get("process", [])
                for u in updates:
                    context[u] = None
            except rasa_nlu.components.MissingArgumentError as e:
                raise Exception("Failed to validate at component '{}'. {}".format(component.name, e.message))

    def train(self, data):
        # type: (TrainingData) -> Interpreter
        """Trains the underlying pipeline by using the provided training data."""

        self.training_data = data

        context = {}

        for component in self.pipeline:
            args = rasa_nlu.components.fill_args(component.pipeline_init_args(), context, self.config.as_dict())
            updates = component.pipeline_init(*args)
            if updates:
                context.update(updates)

        init_context = context.copy()

        context["training_data"] = data

        for component in self.pipeline:
            args = rasa_nlu.components.fill_args(component.train_args(), context, self.config.as_dict())
            updates = component.train(*args)
            if updates:
                context.update(updates)

        return Interpreter(self.pipeline, context=init_context, config=self.config.as_dict())

    def persist(self, path, persistor=None, create_unique_subfolder=True):
        # type: (Text, Optional[Persistor], bool) -> Text
        """Persist all components of the pipeline to the passed path. Returns the directory of the persited model."""

        timestamp = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
        metadata = {
            "language": self.config["language"],
            "pipeline": [component.name for component in self.pipeline],
        }

        if create_unique_subfolder:
            dir_name = os.path.join(path, "model_" + timestamp)
            os.makedirs(dir_name)
        else:
            dir_name = path

        metadata.update(self.training_data.persist(dir_name))

        for component in self.pipeline:
            update = component.persist(dir_name)
            if update:
                metadata.update(update)

        Metadata(metadata, dir_name).persist(dir_name)

        if persistor is not None:
            persistor.send_tar_to_s3(dir_name)
        logging.info("Successfully saved model into '{}'".format(os.path.abspath(dir_name)))
        return dir_name


class Interpreter(object):
    """Use a trained pipeline of components to parse text messages"""

    # Defines all attributes (and their default values) that will be returned by `parse`
    default_output_attributes = {"intent": {"name": "", "confidence": 0.0}, "entities": [], "text": ""}

    @staticmethod
    def load(meta, config, component_builder=None):
        # type: (Metadata, RasaNLUConfig, Optional[rasa_nlu.components.ComponentBuilder]) -> Interpreter
        """Load a stored model and its components defined by the provided metadata."""
        context = {"model_dir": meta.model_dir}
        if component_builder is None:
            # If no builder is passed, every interpreter creation will result in a new builder.
            # hence, no components are reused.
            component_builder = rasa_nlu.components.ComponentBuilder()

        model_config = config.as_dict()
        model_config.update(meta.metadata)

        pipeline = []

        for component_name in meta.pipeline:
            component = component_builder.load_component(component_name, context, model_config, meta)
            try:
                args = rasa_nlu.components.fill_args(component.pipeline_init_args(), context, model_config)
                updates = component.pipeline_init(*args)
                if updates:
                    context.update(updates)
                pipeline.append(component)
            except rasa_nlu.components.MissingArgumentError as e:
                raise Exception("Failed to initialize component '{}'. {}".format(component.name, e.message))

        return Interpreter(pipeline, context, model_config)

    def __init__(self, pipeline, context, config, meta=None):
        # type: (List[Component], Dict[Text, Any], Dict[Text, Any], Optional[Metadata]) -> None

        self.pipeline = pipeline
        self.context = context
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
            return self.default_output_attributes.copy()

        current_context = self.context.copy()

        current_context.update({
            "text": text,
        })

        for component in self.pipeline:
            try:
                args = rasa_nlu.components.fill_args(component.process_args(), current_context, self.config)
                updates = component.process(*args)
                if updates:
                    current_context.update(updates)
            except rasa_nlu.components.MissingArgumentError as e:
                raise Exception("Failed to parse at component '{}'. {}".format(component.name, e.message))

        result = self.default_output_attributes.copy()
        all_attributes = list(self.default_output_attributes.keys()) + self.output_attributes
        # Ensure only keys of `all_attributes` are present and no other keys are returned
        result.update({key: current_context[key] for key in all_attributes if key in current_context})
        return result
