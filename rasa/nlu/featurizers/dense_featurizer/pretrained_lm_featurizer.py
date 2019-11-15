import logging
import os
import re
import scipy.sparse
from typing import Any, Dict, List, Optional, Text
from rasa.nlu import utils
from rasa.nlu.config import RasaNLUModelConfig
from rasa.nlu.featurizers.featurzier import Featurizer
from rasa.nlu.model import Metadata
from rasa.nlu.training_data import Message, TrainingData
from rasa.nlu.constants import (
    MESSAGE_TEXT_ATTRIBUTE,
    MESSAGE_TOKENS_NAMES,
    MESSAGE_ATTRIBUTES,
    MESSAGE_VECTOR_DENSE_FEATURE_NAMES,
    MESSAGE_INTENT_ATTRIBUTE,
    SPACY_FEATURIZABLE_ATTRIBUTES,
)
import torch
from transformers import *
import numpy as np

logger = logging.getLogger(__name__)

model_dictionary = {
    "bert-base-uncased": BertModel,
    "openai-gpt": OpenAIGPTModel,
    "gpt2": GPT2Model,
    "transfo-xl-wt103": TransfoXLModel,
    "xlnet-base-cased": XLNetModel,
    "xlm-mlm-enfr-1024": XLMModel,
    "distilbert-base-uncased": DistilBertModel,
    "roberta-base": RobertaModel,
}

tokenizer_dictionary = {
    "bert-base-uncased": BertTokenizer,
    "openai-gpt": OpenAIGPTTokenizer,
    "gpt2": GPT2Tokenizer,
    "transfo-xl-wt103": TransfoXLTokenizer,
    "xlnet-base-cased": XLNetTokenizer,
    "xlm-mlm-enfr-1024": XLMTokenizer,
    "distilbert-base-uncased": DistilBertTokenizer,
    "roberta-base": RobertaTokenizer,
}

special_tokens_present = {
    "bert-base-uncased": True,
    "openai-gpt": False,
    "gpt2": False,
    "transfo-xl-wt103": False,
    "xlnet-base-cased": True,
    "xlm-mlm-enfr-1024": True,
    "distilbert-base-uncased": True,
    "roberta-base": True,
}


class PreTrainedLMFeaturizer(Featurizer):

    provides = [
        MESSAGE_VECTOR_DENSE_FEATURE_NAMES[attribute]
        for attribute in SPACY_FEATURIZABLE_ATTRIBUTES
    ]

    defaults = {
        # model key identified by HF Transformers
        "model_key": "bert-base-uncased"
    }

    def _load_transformers_params(self):

        self.model_key = self.component_config["model_key"]

        if self.model_key not in tokenizer_dictionary:
            logger.error("{} not a valid model key name".format(self.model_key))
            raise

        logger.info("Loading Tokenizer and Model for {}".format(self.model_key))
        self.tokenizer = tokenizer_dictionary[self.model_key].from_pretrained(
            self.model_key
        )
        self.model = model_dictionary[self.model_key].from_pretrained(self.model_key)
        self.contains_special_token = special_tokens_present[self.model_key]
        self.model.eval()

    def __init__(self, component_config: Dict[Text, Any] = None) -> None:

        super(PreTrainedLMFeaturizer, self).__init__(component_config)

        self._load_transformers_params()

    def train(
        self,
        training_data: TrainingData,
        config: Optional[RasaNLUModelConfig],
        **kwargs: Any,
    ) -> None:

        bs = 128

        for attribute in [MESSAGE_TEXT_ATTRIBUTE]:

            start_index = 0

            while start_index < len(training_data.intent_examples):

                end_index = min(start_index + bs, len(training_data.intent_examples))
                batch_examples = training_data.intent_examples[start_index:end_index]
                batch_text = [ex.get(attribute) for ex in batch_examples]

                batch_feats = self._compute_features(batch_text)

                for index, ex in enumerate(batch_examples):

                    ex.set(
                        MESSAGE_VECTOR_DENSE_FEATURE_NAMES[attribute],
                        self._combine_with_existing_dense_features(
                            ex,
                            batch_feats[index],
                            MESSAGE_VECTOR_DENSE_FEATURE_NAMES[attribute],
                        ),
                    )

                start_index += bs

        # for example in training_data.intent_examples:
        #     for attribute in SPACY_FEATURIZABLE_ATTRIBUTES:
        #         self._set_lm_features(example, attribute)

    def _compute_input_ids(self, batch_examples):

        batch_input_ids = []
        max_seq_len = 0
        actual_seq_lengths = []
        for example in batch_examples:

            example_input_ids = self.tokenizer.encode(
                example, add_special_tokens=self.contains_special_token
            )
            max_seq_len = max(max_seq_len, len(example_input_ids))
            actual_seq_lengths.append(len(example_input_ids))
            batch_input_ids.append(example_input_ids)

        # add padding
        padded_input_ids = []
        for example_input_ids in batch_input_ids:
            padded_input_ids.append(
                example_input_ids
                + [self.tokenizer.pad_token_id] * (max_seq_len - len(example_input_ids))
            )

        return torch.tensor(padded_input_ids), actual_seq_lengths

    def _compute_attention_mask(self, actual_seq_lengths):

        attention_mask = []
        max_seq_length = max(actual_seq_lengths)
        for index in range(len(actual_seq_lengths)):
            example_seq_length = actual_seq_lengths[index]
            attention_mask.append(
                [1] * example_seq_length + [0] * (max_seq_length - example_seq_length)
            )

        attention_mask = np.array(attention_mask).astype(np.float32)

        return torch.tensor(attention_mask)

    def _compute_features(self, batch_inputs):

        batch_model_inputs, actual_seq_lengths = self._compute_input_ids(batch_inputs)
        batch_attention_mask = self._compute_attention_mask(actual_seq_lengths)

        with torch.no_grad():
            last_hidden_states = self.model(
                batch_model_inputs, attention_mask=batch_attention_mask
            )[
                0
            ].numpy()  # Models outputs are now numpy array
            sequence_embedding = last_hidden_states  # First element of batch

            truncated_embeds = self._resolve_special_tokens_in_embeddings(
                sequence_embedding, actual_seq_lengths
            )

            return truncated_embeds

    def _resolve_special_tokens_in_embeddings(self, embeddings, actual_seq_lengths):

        truncated_embeds = []
        for index, embedding in enumerate(embeddings):
            unmasked_embedding = embedding[: actual_seq_lengths[index]]

            if self.contains_special_token:
                # dim - (seq + 2, hdim)
                # Discard SEP token and move CLS token to last index
                unmasked_embedding = unmasked_embedding[:-1, :]  # Discard SEP
                unmasked_embedding = np.roll(
                    unmasked_embedding, -1, axis=0
                )  # Move CLS to back
            else:
                unmasked_embedding = np.concatenate(
                    [unmasked_embedding, np.zeros((1, unmasked_embedding.shape[-1]))],
                    axis=0,
                )
            truncated_embeds.append(unmasked_embedding)

        return np.array(truncated_embeds)

    def _set_lm_features(self, example, attribute=MESSAGE_TEXT_ATTRIBUTE):

        message_attribute_text = example.get(attribute)
        if message_attribute_text:
            # Encode text
            input_ids = torch.tensor(
                [
                    self.tokenizer.encode(
                        message_attribute_text,
                        add_special_tokens=self.contains_special_token,
                    )
                ]
            )  # Add special tokens takes care of adding [CLS], [SEP], <s>... tokens in the right way for each model.
            with torch.no_grad():
                last_hidden_states = self.model(input_ids)[
                    0
                ].numpy()  # Models outputs are now numpy array
                sequence_embedding = last_hidden_states[0]  # First element of batch

                if self.contains_special_token:
                    # dim - (seq + 2, hdim)
                    # Discard SEP token and move CLS token to last index
                    sequence_embedding = sequence_embedding[:-1]  # Discard SEP
                    sequence_embedding = np.roll(
                        sequence_embedding, -1
                    )  # Move CLS to back
                else:
                    sequence_embedding = np.concatenate(
                        [
                            sequence_embedding,
                            np.zeros((1, sequence_embedding.shape[-1])),
                        ],
                        axis=0,
                    )

                features = self._combine_with_existing_dense_features(
                    example,
                    sequence_embedding,
                    MESSAGE_VECTOR_DENSE_FEATURE_NAMES[attribute],
                )
                example.set(MESSAGE_VECTOR_DENSE_FEATURE_NAMES[attribute], features)

    def process(self, message: Message, **kwargs: Any) -> None:

        feats = self._compute_features([message.get(MESSAGE_TEXT_ATTRIBUTE)])
        message.set(
            MESSAGE_VECTOR_DENSE_FEATURE_NAMES[MESSAGE_TEXT_ATTRIBUTE],
            self._combine_with_existing_dense_features(
                message,
                feats[0],
                MESSAGE_VECTOR_DENSE_FEATURE_NAMES[MESSAGE_TEXT_ATTRIBUTE],
            ),
        )
