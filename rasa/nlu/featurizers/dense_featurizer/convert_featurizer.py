import logging
import re
from rasa.nlu.featurizers.featurzier import Featurizer
from typing import Any, Dict, Optional, Text
from rasa.nlu.config import RasaNLUModelConfig
from rasa.nlu.training_data import Message, TrainingData
from rasa.nlu.constants import (
    MESSAGE_TEXT_ATTRIBUTE,
    MESSAGE_VECTOR_DENSE_FEATURE_NAMES,
    SPACY_FEATURIZABLE_ATTRIBUTES,
)
import numpy as np
import tensorflow as tf
import tensorflow_hub as tfhub

logger = logging.getLogger(__name__)


class ConvertFeaturizer(Featurizer):
    provides = [
        MESSAGE_VECTOR_DENSE_FEATURE_NAMES[attribute]
        for attribute in SPACY_FEATURIZABLE_ATTRIBUTES
    ]

    defaults = {
        # model key identified by HF Transformers
        "return_sequence": True
    }

    def _load_model(self):

        self.return_sequence = self.component_config["return_sequence"]

        self.graph = tf.Graph()
        model_url = "http://models.poly-ai.com/convert/v1/model.tar.gz"

        with self.graph.as_default():
            self.session = tf.Session()
            self.module = tfhub.Module(model_url)

            self.text_placeholder = tf.placeholder(dtype=tf.string, shape=[None])
            if self.return_sequence:
                self.sequence_encoding_tensor = self.module(
                    self.text_placeholder, signature="encode_sequence", as_dict=True
                )
                self.tokenized = self.module(
                    self.text_placeholder, signature="tokenize"
                )
            self.sentence_encoding_tensor = self.module(self.text_placeholder)
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

        for attribute in [MESSAGE_TEXT_ATTRIBUTE]:

            start_index = 0

            while start_index < len(training_data.intent_examples):

                end_index = min(start_index + bs, len(training_data.intent_examples))
                batch_examples = training_data.intent_examples[start_index:end_index]

                batch_text = [
                    self._clean_text(ex.get(attribute)) for ex in batch_examples
                ]

                batch_feats = self._compute_features(batch_text)

                for index, ex in enumerate(batch_examples):

                    # print(batch_text[index], batch_feats[index].shape)
                    ex.set(
                        MESSAGE_VECTOR_DENSE_FEATURE_NAMES[attribute],
                        self._combine_with_existing_dense_features(
                            ex,
                            batch_feats[index],
                            MESSAGE_VECTOR_DENSE_FEATURE_NAMES[attribute],
                        ),
                    )

                start_index += bs

    @staticmethod
    def _clean_text(text):

        cleaned_text = re.sub(
            # there is a space or an end of a string after it
            r"[^\w#@&]+(?=\s|$)|"
            # there is a space or beginning of a string before it
            # not followed by a number
            r"(\s|^)[^\w#@&]+(?=[^0-9\s])|"
            # not in between numbers and not . or @ or & or - or #
            # e.g. 10'000.00 or blabla@gmail.com
            # and not url characters
            r"(?<=[^0-9\s])[^\w._~:/?#\[\]()@!$&*+,;=-]+(?=[^0-9\s])",
            " ",
            text,
        )

        # remove multiple occurences of ' '
        cleaned_text = re.sub(" +", " ", cleaned_text)

        if not cleaned_text.strip():
            cleaned_text = text

        return cleaned_text.strip()

    def _tokenize(self, sentence):

        return self.session.run(
            self.tokenized, feed_dict={self.text_placeholder: [sentence]}
        )

    def _compute_features(self, batch_examples):

        sentence_encodings = self.session.run(
            self.sentence_encoding_tensor,
            feed_dict={self.text_placeholder: batch_examples},
        )

        # convert them to a sequence
        sentence_encodings = np.reshape(
            sentence_encodings, (len(batch_examples), 1, -1)
        )

        if self.return_sequence:

            final_embeddings = []

            batch_tokenized = [self._tokenize(sentence) for sentence in batch_examples]

            actual_lens = [token_vector.shape[1] for token_vector in batch_tokenized]

            sequence_encodings = self.session.run(
                self.sequence_encoding_tensor,
                feed_dict={self.text_placeholder: batch_examples},
            )["sequence_encoding"]

            for index in range(len(batch_examples)):

                seq_len = actual_lens[index]
                seq_enc = sequence_encodings[index][:seq_len]
                sent_enc = sentence_encodings[index]

                # tile seq enc to duplicate
                seq_enc = np.tile(seq_enc, (1, 2))

                # add sent_enc to the end
                seq_enc = np.concatenate([seq_enc, sent_enc], axis=0)

                final_embeddings.append(seq_enc)

            return final_embeddings

        return sentence_encodings

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

        feats = self._compute_features(
            [self._clean_text(message.get(MESSAGE_TEXT_ATTRIBUTE))]
        )[0]

        message.set(
            MESSAGE_VECTOR_DENSE_FEATURE_NAMES[MESSAGE_TEXT_ATTRIBUTE],
            self._combine_with_existing_dense_features(
                message,
                feats,
                MESSAGE_VECTOR_DENSE_FEATURE_NAMES[MESSAGE_TEXT_ATTRIBUTE],
            ),
        )
        # self._set_lm_features(message)
