import logging
import os
import re
import scipy.sparse
from typing import Any, Dict, List, Optional, Text
from rasa.nlu import utils
from rasa.nlu.config import RasaNLUModelConfig
from rasa.nlu.tokenizers.tokenizer import Tokenizer, Token
from rasa.nlu.model import Metadata
from rasa.nlu.training_data import Message, TrainingData
from rasa.nlu.constants import (
    MESSAGE_TEXT_ATTRIBUTE,
    MESSAGE_TOKENS_NAMES,
    MESSAGE_ATTRIBUTES,
    MESSAGE_INTENT_ATTRIBUTE,
    SPACY_FEATURIZABLE_ATTRIBUTES,
)
import torch
from transformers import *
import numpy as np
import tensorflow as tf
import tensorflow_hub as tfhub

logger = logging.getLogger(__name__)


class ConvertTokenizer(Tokenizer):

    provides = [
        MESSAGE_TOKENS_NAMES[attribute] for attribute in SPACY_FEATURIZABLE_ATTRIBUTES
    ]

    defaults = {
        # model key identified by HF Transformers
        "use_cls_token": True
    }

    def _load_tokenizer_params(self):

        self.graph = tf.Graph()
        model_url = "http://models.poly-ai.com/convert/v1/model.tar.gz"

        with self.graph.as_default():
            self.session = tf.Session()
            self.module = tfhub.Module(model_url)

            self.text_placeholder = tf.placeholder(dtype=tf.string, shape=[None])
            self.tokenized = self.module(self.text_placeholder, signature="tokenize")

            self.session.run(tf.tables_initializer())
            self.session.run(tf.global_variables_initializer())

    def __init__(self, component_config: Dict[Text, Any] = None) -> None:

        super(ConvertTokenizer, self).__init__(component_config)

        self._load_tokenizer_params()

    def train(
        self,
        training_data: TrainingData,
        config: Optional[RasaNLUModelConfig],
        **kwargs: Any,
    ) -> None:

        for example in training_data.intent_examples:
            for attribute in SPACY_FEATURIZABLE_ATTRIBUTES:
                example.set(
                    MESSAGE_TOKENS_NAMES[attribute],
                    self._get_lm_tokens(example, attribute),
                )

    def _tokenize(self, sentence):

        return self.session.run(
            self.tokenized, feed_dict={self.text_placeholder: [sentence]}
        )

    def _get_lm_tokens(self, example, attribute=MESSAGE_TEXT_ATTRIBUTE):

        message_attribute_text = example.get(attribute)
        if message_attribute_text:

            expanded_tokens_list = []

            # We assume that whitespace tokenizer was used before this and hence tokens attribute is set.
            space_tokens_list = example.get(MESSAGE_TOKENS_NAMES[attribute])

            for token in space_tokens_list:

                token_start, token_end, token_text = token.offset, token.end, token.text

                # Encode text

                split_token_strings = self._tokenize(token_text)[0]

                # print(split_token_strings)

                split_token_strings = [
                    string.decode("utf-8") for string in split_token_strings
                ]

                # print(token_text, split_token_strings)

                current_token_offset = token_start
                for index, string in enumerate(split_token_strings):
                    if index == 0:
                        if index == len(split_token_strings) - 1:
                            s_token_end = token_end
                        else:
                            s_token_end = current_token_offset + len(string)
                        expanded_tokens_list.append(
                            Token(string, token_start, end=s_token_end)
                        )
                    elif index == len(split_token_strings) - 1:
                        expanded_tokens_list.append(
                            Token(string, current_token_offset, end=token_end)
                        )
                    else:
                        expanded_tokens_list.append(
                            Token(
                                string,
                                current_token_offset,
                                end=current_token_offset + len(string),
                            )
                        )
                    current_token_offset += len(string)

            expanded_tokens_list = self.add_cls_token(expanded_tokens_list, attribute)

            # print(message_attribute_text, len(space_tokens_list), len(expanded_tokens_list))

            return expanded_tokens_list

    def process(self, message: Message, **kwargs: Any) -> None:

        tokens = self._get_lm_tokens(message)
        message.set(MESSAGE_TOKENS_NAMES[MESSAGE_TEXT_ATTRIBUTE], tokens)
