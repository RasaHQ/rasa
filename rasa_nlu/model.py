import datetime
import json
import logging
import os

from typing import Optional

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
        # type: (str) -> 'Metadata'
        """Loads the metadata from a models directory."""

        with open(os.path.join(model_dir, 'metadata.json'), 'rb') as f:
            data = json.loads(f.read().decode('utf-8'))
        return Metadata(data, model_dir)

    def __init__(self, metadata, model_dir):
        # type: (dict, Optional[str]) -> None

        self.metadata = metadata
        self.model_dir = model_dir

    def __prepend_path(self, prop):
        if self.metadata.get(prop) is not None:
            return os.path.normpath(os.path.join(self.model_dir, self.metadata[prop]))
        else:
            return None

    @property
    def language(self):
        # type: () -> Optional[str]
        """Language of the underlying model"""

        return self.metadata.get('language')

    @property
    def pipeline(self):
        # type: () -> [str]
        """Names of the processing pipeline elements."""

        if 'pipeline' in self.metadata:
            return self.metadata['pipeline']
        elif 'backend' in self.metadata:   # This is for backwards compatibility of models trained before 0.8
            from rasa_nlu import registry
            return registry.registered_pipeline_templates.get(self.metadata.get('backend'))
        else:
            return []

    def persist(self, model_dir):
        # type: (str) -> None
        """Persists the metadata of a model to a given directory."""

        metadata = self.metadata.copy()

        metadata.update({
            "trained_at": datetime.datetime.now().strftime('%Y%m%d-%H%M%S'),
            "rasa_nlu_version": rasa_nlu.__version__,
        })

        with open(os.path.join(model_dir, 'metadata.json'), 'w') as f:
            f.write(json.dumps(metadata, indent=4))


class Trainer(object):
    """Given a pipeline specification and configuration this trainer will load the data and train all components."""

    # Officially supported languages (others might be used, but might fail)
    SUPPORTED_LANGUAGES = ["de", "en"]

    def __init__(self, config):
        from rasa_nlu.registry import get_component_class

        self.config = config
        self.training_data = None
        self.pipeline = []

        # Transform the passed names of the pipeline components into classes
        for component_name in config.pipeline:
            component_class = get_component_class(component_name)
            if component_class is not None:
                self.pipeline.append(component_class())
            else:
                raise Exception("Unregistered component '{}'. Failed to start trainer.".format(component_name))

    def validate(self):
        # type: () -> None
        """Validates a pipeline before it is run. Ensures, that all arguments are present to train the pipeline."""

        # Validate the init phase
        context = {}

        for component in self.pipeline:
            try:
                rasa_nlu.components.fill_args(component.pipeline_init_args(), context, self.config)
                updates = component.context_provides.get("pipeline_init", [])
                for u in updates:
                    context[u] = None
            except rasa_nlu.components.MissingArgumentError as e:
                raise Exception("Failed to validate at component '{}'. {}".format(component.name, e.message))

        after_init_context = context.copy()

        context["training_data"] = None     # Prepare context for testing the training phase

        for component in self.pipeline:
            try:
                rasa_nlu.components.fill_args(component.train_args(), context, self.config)
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
                rasa_nlu.components.fill_args(component.process_args(), context, self.config)
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
            args = rasa_nlu.components.fill_args(component.pipeline_init_args(), context, self.config)
            updates = component.pipeline_init(*args)
            if updates:
                context.update(updates)

        init_context = context.copy()

        context["training_data"] = data

        for component in self.pipeline:
            args = rasa_nlu.components.fill_args(component.train_args(), context, self.config)
            updates = component.train(*args)
            if updates:
                context.update(updates)

        return Interpreter(self.pipeline, context=init_context, config=self.config)

    def persist(self, path, persistor=None, create_unique_subfolder=True):
        # type: (str, Optional[Persistor], bool) -> str
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
        logging.info("Successfully saved model into '{}'".format(dir_name))
        return dir_name


class Interpreter(object):
    """Use a trained pipeline of components to parse text messages"""

    # Defines all attributes (and their default values) that will be returned by `parse`
    default_output_attributes = {"intent": None, "entities": [], "text": ""}

    @staticmethod
    def load(meta, rasa_config):
        # type: (Metadata, RasaNLUConfig) -> Interpreter
        """Load a stored model and its components defined by the provided metadata."""
        from rasa_nlu.registry import load_component_by_name

        context = {"model_dir": meta.model_dir}

        config = dict(rasa_config.items())
        config.update(meta.metadata)

        pipeline = []

        for component_name in meta.pipeline:
            try:
                component = load_component_by_name(component_name, context, config)
            except rasa_nlu.components.MissingArgumentError as e:
                raise Exception("Failed to create/load component '{}'. {}".format(component_name, e.message))
            try:
                rasa_nlu.components.init_component(component, context, config)
                pipeline.append(component)
            except rasa_nlu.components.MissingArgumentError as e:
                raise Exception("Failed to initialize component '{}'. {}".format(component.name, e.message))

        return Interpreter(pipeline, context, config, meta)

    def __init__(self, pipeline, context, config, meta=None):
        # type: ([Component], dict, dict, Optional[Metadata]) -> None

        self.pipeline = pipeline
        self.context = context
        self.config = config
        self.meta = meta
        self.output_attributes = [output for component in pipeline for output in component.output_provides]

    def parse(self, text):
        # type: (basestring) -> dict
        """Parse the input text, classify it and return an object containing its intent and entities."""

        if type(text) is str:
            text = unicode(text, "utf-8")

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
        all_attributes = self.default_output_attributes.keys() + self.output_attributes
        # Ensure only keys of `all_attributes` are present and no other keys are returned
        result.update({key: current_context[key] for key in all_attributes if key in current_context})
        return result
