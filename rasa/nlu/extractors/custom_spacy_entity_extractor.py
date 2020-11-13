import os

import spacy
from spacy.training.iob_utils import biluo_tags_from_offsets

from rasa.nlu.extractors.extractor import EntityExtractor
from rasa.nlu.model import Metadata, InvalidModelError

from typing import Any, Dict, List, Text, Optional, Type
from rasa.shared.nlu.constants import ENTITIES, TEXT, INTENT, METADATA


class SpacyCustomNER(EntityExtractor):
    """A custom sentiment analysis component"""
    name = "custom_entities"
    provides = ["custom_entities"]
    language_list = ["en"]
    print('initialised the class')

    defaults = {
        "custom_model": None,
        "project": None,
        # by default all dimensions recognized by spacy are returned
        # dimensions can be configured to contain an array of strings
        # with the names of the dimensions to filter for
        "dimensions": None
    }

    def __init__(
            self, component_config: Dict[Text, Any] = None, nlp: "Language" = None
    ) -> None:
        self.nlp = nlp
        super(SpacyCustomNER, self).__init__(component_config)

    @staticmethod
    def load_model(spacy_model_name: Text) -> "Language":
        """Try loading the model, catching the OSError if missing."""
        import spacy

        try:
            return spacy.load(spacy_model_name, disable=["parser"])
        except OSError:
            raise InvalidModelError(
                "Model '{}' is not a linked spaCy model.  "
                "Please download and/or link a spaCy model, "
                "e.g. by running:\npython -m spacy download "
                "en_core_web_md\npython -m spacy link "
                "en_core_web_md en".format(spacy_model_name)
            )

    def train(self, training_data, cfg, **kwargs):
        """Load the sentiment polarity labels from the text
           file, retrieve training tokens and after formatting
           data train the classifier."""
        from spacy.cli.project.run import project_run
        from spacy.cli.project.assets import project_assets
        from pathlib import Path

        # root = Path(__file__).parent
        nlp = spacy.blank(cfg.language)

        component_config = None
        for pipe in cfg.pipeline:
            if pipe['name'] == str(self.__class__.name):
                component_config = pipe
                break
        if component_config.get('project'):
            root = Path(component_config.get('project'))
            print('Project path from config:{}'.format(root.name))
        else:
            root = Path(os.path.abspath(os.getcwd()))
        if training_data.entity_examples:
            with open(os.path.join(root, 'ner/data/data.iob'), 'w') as fout:
                for example in training_data.entity_examples:
                    entities = []
                    doc = nlp(example.get(TEXT))
                    for ent in example.get(ENTITIES):
                        entities.append((ent.get('start'), ent.get('end'), ent.get('entity')))

                    tags = biluo_tags_from_offsets(doc, entities)
                    tokens = [token.text for token in doc]
                    ner_training = [tok + '|' + tag for tok, tag in zip(tokens, tags)]
                    fout.write(' '.join(ner_training) + '\n')

            project_assets(root)
            project_run(root, "all")

        else:
            raise Exception(
                "No example with Entities to train NER model: {}".format(SpacyCustomNER)
            )

    def convert_to_rasa(self, value, confidence=None):
        """Convert model output into the Rasa NLU compatible output format."""
        ent_list = []
        for ent in value.ents:
            ent_dict = {
                "entity": ent.label_,
                "value": ent.text,
                "start": ent.start,
                "end": ent.end,
                "confidence": confidence,
                "extractor": "CustomEntityExtractor"
            }
            ent_list.append(ent_dict)

        return ent_list

    def preprocessing(self, tokens):
        """Create bag-of-words representation of the training examples."""
        return ({word: True for word in tokens})

    @staticmethod
    def extract_entities(doc: "Doc") -> List[Dict[Text, Any]]:
        entities = [
            {
                "entity": ent.label_,
                "value": ent.text,
                "start": ent.start_char,
                "confidence": None,
                "end": ent.end_char,
            }
            for ent in doc.ents
        ]
        return entities

    def process(self, message, **kwargs):
        """Retrieve the tokens of the new message, pass it to the classifier
            and append prediction results to the message class."""
        doc = self.nlp(message.get(TEXT))
        all_extracted = self.add_extractor_name(self.extract_entities(doc))
        dimensions = self.component_config["dimensions"]
        extracted = SpacyCustomNER.filter_irrelevant_entities(
            all_extracted, dimensions
        )
        message.set(ENTITIES, message.get(ENTITIES, []) + extracted, add_to_output=True)
        # entity = self.convert_to_rasa(doc)
        # message.set(ENTITIES, entity, add_to_output=True)

    def persist(self, file_name, model_dir):
        """Pass because a pre-trained model is already persisted"""
        pass

    @classmethod
    def load(
            self,
            meta: Dict[Text, Any],
            model_dir: Text = None,
            model_metadata: "Metadata" = None,
            cached_component: Optional["SpacyCustomNER"] = None,
            **kwargs: Any,
    ) -> "SpacyCustomNER":
        model_path = meta.get("custom_model")
        nlp = self.load_model(model_path)
        self.ensure_proper_language_model(nlp)
        return self(meta, nlp)

    @staticmethod
    def ensure_proper_language_model(nlp: Optional["Language"]) -> None:
        """Checks if the spacy language model is properly loaded.

        Raises an exception if the model is invalid."""

        if nlp is None:
            raise Exception(
                "Failed to load spacy language model. "
                "Loading the model returned 'None'."
            )
        if nlp.path is None:
            # Spacy sets the path to `None` if
            # it did not load the model from disk.
            # In this case `nlp` is an unusable stub.
            raise Exception(
                "Failed to load spacy language model for "
                "lang '{}'. Make sure you have downloaded the "
                "correct model (https://spacy.io/docs/usage/)."
                "".format(nlp.lang)
            )
