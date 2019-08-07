import logging
import os
import re
from typing import Any, Dict, List, Optional, Text

from rasa.nlu import utils
from rasa.nlu.config import RasaNLUModelConfig
from rasa.nlu.featurizers import Featurizer
from rasa.nlu.model import Metadata
from rasa.nlu.training_data import Message, TrainingData
import numpy as np

logger = logging.getLogger(__name__)


class DummyNERFeaturizer(Featurizer):
    """Dummy featurizer to test the custom_ner_features functionality
    """

    provides = ["ner_features"]

    requires = ["tokens"]

    defaults = {
        # limit vocabulary size
        "num_features": 10  # int or None
    }

    def __init__(
        self,
        component_config: Dict[Text, Any] = None,
        vectorizer: Optional["CountVectorizer"] = None,
    ) -> None:
        """Construct a new count vectorizer using the sklearn framework."""

        super(DummyNERFeaturizer, self).__init__(component_config)
        self.num_features = self.component_config["num_features"]

    def train(
        self, training_data: TrainingData, cfg: RasaNLUModelConfig = None, **kwargs: Any
    ) -> None:
        """Train the featurizer.
        """
        for i, example in enumerate(training_data.intent_examples):
            # create bag for each example
            self.process(example)

    def process(self, message: Message, **kwargs: Any) -> None:
        tokens = message.get("tokens", [])
        ner_features = np.random.rand(len(tokens), self.num_features)
        message.set("ner_features", ner_features)

    def persist(self, file_name: Text, model_dir: Text) -> Optional[Dict[Text, Any]]:
        """Persist this model into the passed directory.

        Returns the metadata necessary to load the model again.
        """
        return {"file": file_name}

    @classmethod
    def load(
        cls,
        meta: Dict[Text, Any],
        model_dir: Text = None,
        model_metadata: Metadata = None,
        cached_component: Optional["DummyNERFeaturizer"] = None,
        **kwargs: Any
    ) -> "DummyNERFeaturizer":
        return cls(meta)
