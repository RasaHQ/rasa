import datetime
import json
import logging
import os

from typing import Optional

import rasa_nlu.components
from rasa_nlu.components import Component
from rasa_nlu.config import RasaNLUConfig


class InvalidModelError(Exception):
    """Raised when a model failed to load.

    Attributes:
        message -- explanation of why the model is invalid
    """

    def __init__(self, message):
        self.message = message


class Metadata(object):
    @staticmethod
    def load(model_dir):
        with open(os.path.join(model_dir, 'metadata.json'), 'rb') as f:
            data = json.loads(f.read().decode('utf-8'))
        return Metadata(data, model_dir)

    def __init__(self, metadata, model_dir):
        self.metadata = metadata
        self.model_dir = model_dir

    def __prepend_path(self, prop):
        if self.metadata.get(prop) is not None:
            return os.path.normpath(os.path.join(self.model_dir, self.metadata[prop]))
        else:
            return None

    @property
    def feature_extractor_path(self):
        return self.__prepend_path("feature_extractor")

    @property
    def intent_classifier_path(self):
        return self.__prepend_path("intent_classifier")

    @property
    def entity_extractor_path(self):
        return self.__prepend_path("entity_extractor")

    @property
    def entity_synonyms_path(self):
        return self.__prepend_path("entity_synonyms")

    @property
    def language(self):
        return self.metadata.get('language')

    @property
    def pipeline(self):
        if 'pipeline' in self.metadata:
            return self.metadata.get('pipeline')
        elif 'model_template' in self.metadata:
            from rasa_nlu import registry
            return registry.registered_model_templates.get(self.metadata.get('model_template'))
        else:
            return []


class Trainer(object):
    SUPPORTED_LANGUAGES = ["de", "en"]

    def __init__(self, config):
        from rasa_nlu.registry import registered_components

        self.config = config
        self.training_data = None
        self.pipeline = []

        for cp in config.pipeline:
            if cp in registered_components:
                self.pipeline.append(registered_components[cp]())
            else:
                raise Exception("Unregistered component '{}'. Failed to start trainer.".format(cp))

    def validate(self):
        context = {
            "training_data": None,
        }

        for component in self.pipeline:
            try:
                rasa_nlu.components.fill_args(component.train_args(), context, self.config)
                updates = component.context_provides
                for u in updates:
                    context[u] = None
            except rasa_nlu.components.MissingArgumentError as e:
                raise Exception("Failed to validate at component '{}'. {}".format(component.name, e.message))

    def train(self, data):
        self.training_data = data

        context = {}

        for component in self.pipeline:
            args = rasa_nlu.components.fill_args(component.pipeline_init_args(), context, self.config)
            updates = component.pipeline_init(*args)
            if updates:
                context.update(updates)

        context["training_data"] = data

        for component in self.pipeline:
            args = rasa_nlu.components.fill_args(component.train_args(), context, self.config)
            updates = component.train(*args)
            if updates:
                context.update(updates)

    def persist(self, path, persistor=None, create_unique_subfolder=True):
        timestamp = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
        metadata = {}

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

        Trainer.write_training_metadata(dir_name, timestamp, self.pipeline,
                                        self.config["language"], metadata)

        if persistor is not None:
            persistor.send_tar_to_s3(dir_name)
        logging.info("Successfully saved model into '{}'".format(dir_name))
        return dir_name

    @staticmethod
    def write_training_metadata(output_folder, timestamp, pipeline,
                                language, additional_metadata):
        metadata = additional_metadata.copy()

        metadata.update({
            "trained_at": timestamp,
            "language": language,
            "rasa_nlu_version": rasa_nlu.__version__,
            "pipeline": [component.name for component in pipeline]
        })

        with open(os.path.join(output_folder, 'metadata.json'), 'w') as f:
            f.write(json.dumps(metadata, indent=4))


class Interpreter(object):
    output_attributes = {"intent": None, "entities": [], "text": ""}

    @staticmethod
    def load(meta, rasa_config):
        # type: (Metadata, RasaNLUConfig) -> Interpreter

        context = {
            "model_dir": meta.model_dir,
        }

        config = dict(rasa_config.items())
        config.update(meta.metadata)

        pipeline = []

        for component_name in meta.pipeline:
            try:
                component = rasa_nlu.components.load_component_instance(component_name, context, config)
                pipeline.append(component)
            except rasa_nlu.components.MissingArgumentError as e:
                raise Exception("Failed to create/load component '{}'. {}".format(component_name, e.message))

        return Interpreter(pipeline, context, config, meta)

    def __init__(self, pipeline, context, config, meta=None):
        # type: ([Component], dict, dict, Optional[Metadata]) -> None

        self.pipeline = pipeline
        self.context = context
        self.config = config
        self.meta = meta
        self.init_components()

    def init_components(self):
        # type: () -> None

        for component in self.pipeline:
            try:
                rasa_nlu.components.init_component(component, self.context, self.config)
            except rasa_nlu.components.MissingArgumentError as e:
                raise Exception("Failed to initialize component '{}'. {}".format(component.name, e.message))

    def parse(self, text):
        # type: (str) -> dict
        """Parse the input text, classify it and return an object containing its intent and entities."""

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

        result = self.output_attributes.copy()
        # Ensure all keys of `output_attributes` are present and no other keys are returned
        result.update({key: current_context[key] for key in self.output_attributes if key in current_context})
        return result
