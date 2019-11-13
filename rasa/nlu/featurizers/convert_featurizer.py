import logging
import os
import re
import scipy.sparse
from rasa.nlu.featurizers import Featurizer
from typing import Any, Dict, List, Optional, Text
from rasa.nlu import utils
from rasa.nlu.config import RasaNLUModelConfig
from rasa.nlu.model import Metadata
from rasa.nlu.training_data import Message, TrainingData
from rasa.nlu.constants import (
    MESSAGE_TEXT_ATTRIBUTE,
    MESSAGE_TOKENS_NAMES,
    MESSAGE_ATTRIBUTES,
    MESSAGE_INTENT_ATTRIBUTE,
    MESSAGE_VECTOR_FEATURE_NAMES,
    SPACY_FEATURIZABLE_ATTRIBUTES,
)
import numpy as np
import tensorflow as tf
import tensorflow_text
import tensorflow_hub as tfhub

logger = logging.getLogger(__name__)


class ConvertFeaturizer(Featurizer):

    provides = [
        MESSAGE_VECTOR_FEATURE_NAMES[attribute]
        for attribute in SPACY_FEATURIZABLE_ATTRIBUTES
    ]

    def _load_model(self):

        self.graph = tf.Graph()
        model_url = "http://models.poly-ai.com/convert/v1/model.tar.gz"

        with self.graph.as_default():
            self.session = tf.Session()
            self.module = tfhub.Module(model_url)

            self.text_placeholder = tf.placeholder(dtype=tf.string, shape=[None])
            self.encoding_tensor = self.module(self.text_placeholder)
            self.session.run(tf.tables_initializer())
            self.session.run(tf.global_variables_initializer())
            
    def __init__(self, component_config: Dict[Text, Any] = None) -> None:

        super(ConvertFeaturizer, self).__init__(component_config)

        self._load_model()

    def train(
        self,
        training_data: TrainingData,
        config: Optional[RasaNLUModelConfig],
        **kwargs: Any,
    ) -> None:

        bs = 64

        print(len(training_data.intent_examples))

        for attribute in [MESSAGE_TEXT_ATTRIBUTE]:

            start_index = 0

            while start_index < len(training_data.intent_examples):

                if start_index % 100 == 0:
                    print('done', start_index)

                end_index = min(start_index+bs,len(training_data.intent_examples))

                print(start_index,end_index)
                batch_examples = training_data.intent_examples[start_index:end_index]
                batch_text = [ex.get(attribute) for ex in batch_examples]

                batch_feats = self.compute_features(batch_text)

                for index, ex in enumerate(batch_examples):

                    # print(type(batch_feats[index]),batch_feats[index].shape)

                    # if batch_feats[index] is None:
                    # print(batch_text[index])

                    ex.set(MESSAGE_VECTOR_FEATURE_NAMES[attribute], batch_feats[index])
                    # ex.set(MESSAGE_VECTOR_FEATURE_NAMES[attribute],self._combine_with_existing_features(ex,batch_feats[index], MESSAGE_VECTOR_FEATURE_NAMES[attribute]))

                start_index += bs

            '''for index,example in enumerate(training_data.intent_examples):

                if index % 100 == 0:
                    print('done',index)
           
                self._set_lm_features(example, attribute)
            '''

    def compute_features(self, batch_examples):

        return self.session.run(self.encoding_tensor, feed_dict={self.text_placeholder: batch_examples})

    def _set_lm_features(self, example, attribute=MESSAGE_TEXT_ATTRIBUTE):

        message_attribute_text = example.get(attribute)
        if message_attribute_text:
            # Encode text
            features = self.module([message_attribute_text])[0]
            features = self._combine_with_existing_features(
                example, features, MESSAGE_VECTOR_FEATURE_NAMES[attribute]
            )
            # print(features.shape)
            example.set(MESSAGE_VECTOR_FEATURE_NAMES[attribute], features)

    def process(self, message: Message, **kwargs: Any) -> None:

        feats = self.compute_features([message.get(MESSAGE_TEXT_ATTRIBUTE)])[0]
        message.set(MESSAGE_VECTOR_FEATURE_NAMES[MESSAGE_TEXT_ATTRIBUTE],feats)
        # self._set_lm_features(message)
