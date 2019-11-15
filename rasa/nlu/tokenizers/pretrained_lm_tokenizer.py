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

logger = logging.getLogger(__name__)

tokenizer_dictionary = {
    "bert-base-uncased": BertTokenizer,
    "openai-gpt": OpenAIGPTTokenizer,
    # "gpt2": GPT2Tokenizer,
    # "transfo-xl-wt103": TransfoXLTokenizer,
    # "xlnet-base-cased": XLNetTokenizer,
    # "xlm-mlm-enfr-1024": XLMTokenizer,
    # "distilbert-base-uncased": DistilBertTokenizer,
    # "roberta-base": RobertaTokenizer,
}

special_tokens_present = {
    "bert-base-uncased": True,
    "openai-gpt": False,
    # "gpt2": False,
    # "transfo-xl-wt103": False,
    # "xlnet-base-cased": True,
    # "xlm-mlm-enfr-1024": True,
    # "distilbert-base-uncased": True,
    # "roberta-base": True,
}


class PreTrainedLMTokenizer(Tokenizer):

    provides = [
        MESSAGE_TOKENS_NAMES[attribute] for attribute in SPACY_FEATURIZABLE_ATTRIBUTES
    ]

    defaults = {
        # model key identified by HF Transformers
        "use_cls_token": True,
        "lm_key": "bert-base-uncased",
    }

    def _load_tokenizer_params(self):

        self.lm_key = self.component_config["lm_key"]

        if self.lm_key not in tokenizer_dictionary:
            logger.error("{} not a valid model key name".format(self.lm_key))
            raise

        logger.info("Loading Tokenizer for {}".format(self.lm_key))
        self.tokenizer = tokenizer_dictionary[self.lm_key].from_pretrained(self.lm_key)
        self.contains_special_token = special_tokens_present[self.lm_key]

    def __init__(self, component_config: Dict[Text, Any] = None) -> None:

        super(PreTrainedLMTokenizer, self).__init__(component_config)

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

    def _get_lm_tokens(self, example, attribute=MESSAGE_TEXT_ATTRIBUTE):

        message_attribute_text = example.get(attribute)
        if message_attribute_text:

            expanded_tokens_list = []

            # We assume that whitespace tokenizer was used before this and hence tokens attribute is set.
            space_tokens_list = example.get(MESSAGE_TOKENS_NAMES[attribute])

            for token in space_tokens_list:

                token_start, token_end, token_text = token.offset, token.end, token.text

                # Encode text

                # Add special tokens takes care of adding [CLS], [SEP], <s>... tokens in the right way for each model.
                split_token_ids = self.tokenizer.encode(token_text)

                split_token_strings = self.tokenizer.convert_ids_to_tokens(
                    split_token_ids
                )

                # print(split_token_strings)

                current_token_offset = token_start
                for index, string in enumerate(split_token_strings):
                    if index == 0:
                        expanded_tokens_list.append(
                            Token(
                                string,
                                token_start,
                                end=current_token_offset + len(string),
                            )
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
