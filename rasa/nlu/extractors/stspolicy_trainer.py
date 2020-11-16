from rasa.nlu.extractors.extractor import EntityExtractor

from typing import Any, Dict, List, Text, Optional, Type
from rasa.shared.nlu.constants import ENTITIES, TEXT, INTENT, METADATA


class STSFallbackTrainer(EntityExtractor):
    defaults = {
        "host": '127.0.0.1',
        "port": 9502,
        "arch": None,
        "dropout": 0.1,
        "accumulate_gradient": 1,
        "patience": 500,
        "max_epochs": 0,
        "max_steps": 5000,
        "eval_frequency": 200,
    }

    def __init__(
            self, component_config: Dict[Text, Any] = None) -> None:
        self.component_config = component_config
        super(STSFallbackTrainer, self).__init__(component_config)

    def train(self, training_data, cfg, **kwargs):
        """Load the sentiment polarity labels from the text
           file, retrieve training tokens and after formatting
           data train the classifier."""
        intents = []
        texts = []
        for example in training_data.intent_examples:
            texts.append(example.get(TEXT))
            intents.append(example.get(INTENT))
        import requests
        url = 'http://{0}:{1}/train'.format(self.component_config.get('host'),
                                            self.component_config.get('port'))
        data = {"text": texts, "intents": intents, "params": [self.component_config]}
        print(requests.put(url, json=data).text)
