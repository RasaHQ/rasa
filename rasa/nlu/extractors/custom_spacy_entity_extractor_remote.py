import os

import spacy
from spacy.training.iob_utils import biluo_tags_from_offsets

from rasa.nlu.extractors.extractor import EntityExtractor
from rasa.nlu.model import Metadata, InvalidModelError

from typing import Any, Dict, List, Text, Optional, Type
from rasa.shared.nlu.constants import ENTITIES, TEXT, INTENT, METADATA


class RemoteSpacyCustomNER(EntityExtractor):
    """A custom sentiment analysis component"""
    name = "custom_entities"
    provides = ["custom_entities"]
    language_list = ["en"]
    print('initialised the class')

    defaults = {
        "host": None,
        "port": None,
        "arch": None,
        # by default all dimensions recognized by spacy are returned
        # dimensions can be configured to contain an array of strings
        # with the names of the dimensions to filter for
        "dimensions": None
    }

    def __init__(
            self, component_config: Dict[Text, Any] = None, nlp: "Language" = None
    ) -> None:
        self.nlp = nlp
        super(RemoteSpacyCustomNER, self).__init__(component_config)

    def train(self, training_data, cfg, **kwargs):
        """Load the sentiment polarity labels from the text
           file, retrieve training tokens and after formatting
           data train the classifier."""
        import requests
        url = 'http://127.0.0.1:9501/train'
        data = {"text": ['hello', 'training!']}
        print(requests.put(url, json=data).text)

    def process(self, message, **kwargs):
        """Retrieve the tokens of the new message, pass it to the classifier
            and append prediction results to the message class."""
        import requests
        url = 'http://127.0.0.1:9501/predict'
        data = {"text": ['hello', 'prediction!']}
        print(requests.get(url, json=data).text)
        all_extracted = []
        message.set(ENTITIES, message.get(ENTITIES, []) + all_extracted, add_to_output=True)

    def persist(self, file_name, model_dir):
        """Pass because a pre-trained model is already persisted"""
        pass
