from rasa_nlu.featurizers.ngram_featurizer import NGramFeaturizer
from rasa_nlu.utils.mitie_utils import MitieNLP
import datetime
import json
import logging
import os

import rasa_nlu.components
from rasa_nlu.classifiers.mitie_intent_classifier import MitieIntentClassifier
from rasa_nlu.classifiers.simple_intent_classifier import SimpleIntentClassifier
from rasa_nlu.classifiers.sklearn_intent_classifier import SklearnIntentClassifier
from rasa_nlu.extractors.entity_synonyms import EntitySynonymMapper
from rasa_nlu.extractors.mitie_entity_extractor import MitieEntityExtractor
from rasa_nlu.extractors.spacy_entity_extractor import SpacyEntityExtractor
from rasa_nlu.featurizers.mitie_featurizer import MitieFeaturizer
from rasa_nlu.featurizers.spacy_featurizer import SpacyFeaturizer
from rasa_nlu.tokenizers.mitie_tokenizer import MitieTokenizer

from rasa_nlu.utils.spacy_utils import SpacyNLP

component_classes = [
    SpacyNLP, SpacyEntityExtractor, SklearnIntentClassifier, SpacyFeaturizer,
    MitieNLP, MitieEntityExtractor, MitieIntentClassifier, MitieFeaturizer, MitieTokenizer,
    SimpleIntentClassifier, EntitySynonymMapper, NGramFeaturizer]


registered_components = {component.name: component for component in component_classes}


registered_pipelines = {
    "spacy_sklearn": [
        "init_spacy",
        "ner_spacy",
        "ner_synonyms",
        "intent_featurizer_spacy",
        "intent_sklearn",
    ],
    "mitie": [
        "init_mitie",
        "tokenizer_mitie",
        "ner_mitie",
        "ner_synonyms",
        "intent_featurizer_mitie",
        "intent_mitie",
    ],
    "mitie_sklearn": [
        "init_mitie",
        "tokenizer_mitie",
        "ner_mitie",
        "ner_synonyms",
        "intent_featurizer_mitie",
        "intent_sklearn",
    ]
}


class Trainer(object):
    SUPPORTED_LANGUAGES = ["de", "en"]

    def __init__(self, config):
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

        context = {
            "training_data": data,
        }

        for component in self.pipeline:
            args = rasa_nlu.components.fill_args(component.pipeline_init_args(), context, self.config)
            updates = component.pipeline_init(*args)
            if updates:
                context.update(updates)

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
        """
        :type meta: rasa_nlu.model.Metadata
        """
        context = {
            "model_dir": meta.model_dir,
        }

        config = dict(rasa_config.items())
        config.update(meta.metadata)

        pipeline = []

        for component_name in meta.pipeline:
            try:
                component = rasa_nlu.components.load_component_instance(component_name, context, config)
                rasa_nlu.components.init_component(component, context, config)
                pipeline.append(component)
            except rasa_nlu.components.MissingArgumentError as e:
                raise Exception("Failed to initialize component '{}'. {}".format(component_name, e.message))

        return Interpreter(pipeline, context, config, meta)

    def __init__(self, pipeline, context, config, meta=None):
        self.pipeline = pipeline
        self.context = context
        self.config = config
        self.meta = meta

    def parse(self, text):
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
        result.update({key: current_context[key] for key in self.output_attributes if key in current_context})
        return result
