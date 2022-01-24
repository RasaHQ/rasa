from abc import ABC
from dataclasses import dataclass
from typing import Dict, List, Set, Text, Optional
import re

import numpy as np

from rasa.shared.nlu.training_data.message import Message
from rasa.nlu.constants import TOKENS_NAMES
from rasa.nlu.tokenizers.tokenizer import Token
from rasa.shared.nlu.constants import (
    TEXT,
    INTENT,
    ENTITIES,
    ENTITY_ATTRIBUTE_END,
    ENTITY_ATTRIBUTE_START,
    ENTITY_ATTRIBUTE_TYPE,
    ENTITY_ATTRIBUTE_VALUE,
)

from tests.nlu.dummy_data.dummy_features import (
    DummyFeatures,
    ConcatenatedFeaturizations,
    FeaturizerDescription,
)


class DummyData(ABC):
    """A dummy dataset."""

    def create_messages(
        self,
    ) -> List[Message]:
        """Creates some messages."""
        pass


class DummyFeaturizer(ABC):
    """A dummy featurizer."""

    def featurize_messages(
        self,
        messages: List[Message],
    ) -> None:
        pass


@dataclass
class IntentAndEntitiesEncodings:
    """Encodings expected for a specific subset of the data."""

    indices: List[int]  # identifies the subset
    intents_mhot: np.ndarray
    intents_ids: List[int]
    entities_ids: List[int]
    entities_bilou_ids: List[Optional[List[int]]]
    entities_bilou_tags: List[Optional[List[Text]]]  # to test side effects on messages


class TextIntentAndEntitiesDummy(DummyData, DummyFeaturizer):
    """A featurized dataset containing text and (optionally) intent and/or entities.

    The dummy messages have the following properties:
    - Each message contains a text.
    - There are message with: only an intent, only entities, intent+entities, and
      neither intent nor entities.
    - Some messages contain one out of two intents.
    - One of the two intents is tokenized into more than one token.

    The dummy featurization has the following properties:
    - 2 Sparse and 2 dense featurizers generate features.
    - Each featurizer is used to featurize the text and (if requested) the intents.
    - Each featurizer produces sequence features and (if requested) sentence features.
    """

    def __init__(
        self, featurize_intents: bool, no_sentence_features: Optional[Set[Text]]
    ) -> None:
        no_sentence_features = no_sentence_features or set()
        attributes = {TEXT, INTENT} if featurize_intents else {TEXT}
        self.featurizers = [
            FeaturizerDescription(
                name=f"{idx}",
                sequence_dim=idx,
                sentence_dim=idx + 1,
                sentence_attributes=attributes.difference(no_sentence_features),
                sequence_attributes=attributes,
                is_sparse=(idx in [1, 2]),
            )
            for idx in [1, 2, 3, 4]
        ]
        self._rand_featurizer = DummyFeatures(
            featurizer_descriptions=self.featurizers,
        )
        self._num_messages = len(self.create_messages())

    @staticmethod
    def tokenize(text: Text) -> List[Token]:
        return [
            Token(text=match.group(), start=match.start())
            for match in re.finditer(r"\w+", text)
        ]

    def create_messages(
        self,
    ) -> List[Message]:
        """Generates some test messages."""

        # prepare an example with entities
        text = "the city of bielefeld does not exist"
        tokens = self.tokenize(text)
        entities = [
            {
                ENTITY_ATTRIBUTE_VALUE: "city of bielefeld",
                ENTITY_ATTRIBUTE_START: tokens[1].start,
                ENTITY_ATTRIBUTE_END: tokens[3].end,
                ENTITY_ATTRIBUTE_TYPE: "city",
            },
            {
                ENTITY_ATTRIBUTE_VALUE: tokens[-1].text,
                ENTITY_ATTRIBUTE_START: tokens[-1].start,
                ENTITY_ATTRIBUTE_END: tokens[-1].end,
                ENTITY_ATTRIBUTE_TYPE: "what",
            },
        ]

        # create messages
        messages = [
            # 0: only text
            Message(
                data={
                    TEXT: "just a text",
                }
            ),
            # 1: "intent2" (with some added spaces in the intent name)
            Message(
                data={
                    TEXT: "some text that is irrelevant, only tokens count",
                    INTENT: "intent2  ",  # with some added space
                },
                ENTITIES=[],
            ),
            # 2: "intent1 with more tokens"
            Message(
                data={
                    TEXT: "word",
                    INTENT: "intent1 with more tokens",
                },
                ENTITIES=[],
            ),
            # 3: "intent1 with more tokens" + entities
            Message(
                data={
                    TEXT: text,
                    INTENT: "intent1 with more tokens",
                    ENTITIES: entities,
                }
            ),
            # 4: no intent -> core message
            Message(
                data={
                    TEXT: text,
                    INTENT: "",  # this is like no intent
                    ENTITIES: entities,
                }
            ),
        ]

        # Tokenize them all
        for message in messages:
            for attribute in [TEXT, INTENT]:
                if attribute in message.data.keys():
                    message.set(
                        TOKENS_NAMES[attribute],
                        self.tokenize(message.get(attribute)),
                    )

        return messages

    def intent_classifier_usage(
        self,
    ) -> Dict[Text, IntentAndEntitiesEncodings]:
        """Describes how an intent classifier should use this dataset."""

        training = IntentAndEntitiesEncodings(
            # only NLU messages are used
            indices=[1, 2, 3],
            # intents names are sorted lexicographically
            intents_ids=[1, 0, 0],
            intents_mhot=np.array([[[0, 1]], [[1, 0]], [[1, 0]]]),
            entities_ids=[[0], [0], [0, 1, 1, 1, 0, 0, 2]],
            # where 0 = no entity, 1 = city, 2 = what
            entities_bilou_ids=[[0], [0], [0, 1, 2, 3, 0, 0, 8]],
            # where 0 = no entity, 1/2/3/4 = B/I/L/U-city, and 5/6/7/8 = B/I/L/U-what
            entities_bilou_tags=[
                None,
                None,
                [
                    "O",
                    "B-city",
                    "I-city",
                    "L-city",
                    "O",
                    "O",
                    "U-what",
                ],
            ],
        )

        label_data = IntentAndEntitiesEncodings(
            # intents are sorted lexicographically
            indices=[2, 1],
            intents_mhot=np.array([[[1, 0]], [[0, 1]]]),
            intents_ids=[0, 1],
            # entities do not appear in label data
            entities_ids=[],
            entities_bilou_ids=[],
            entities_bilou_tags=[],
        )

        prediction = IntentAndEntitiesEncodings(
            # all messages are used in the given order
            indices=list(range(self._num_messages)),
            # no intents or entities are contained
            intents_mhot=np.array([]),
            intents_ids=[],
            entities_ids=[],
            entities_bilou_ids=[],
            entities_bilou_tags=[],
        )

        return {
            "training": training,
            "label_data": label_data,
            "prediction": prediction,
        }

    def featurize_messages(
        self,
        messages: List[Message],
    ) -> None:
        """Featurizes the given messages."""
        self._rand_featurizer.apply_featurization(
            messages,
        )

    def create_and_concatenate_features(
        self,
        messages: List[Message],
        used_featurizers: List[Text],
        attributes: Optional[Set[Text]] = None,
    ) -> Dict[Text, List[ConcatenatedFeaturizations]]:
        """Creates features for the given messages and concatenates them.

        For each message, the features that have been created using the
        `used_featurizers` that have the same type and sparseness property are
        concatenated along the last dimension.

        Args:
            messages: the messages for which we want to create dummy features; no
               features will be added
            used_featurizers: determines which of the featurizers of this dummy
                featurizer are used to create the concatenated feature matrices
                (this is application dependent and hence not part of the dummy
                dataset definition)
            attributes: attributes for which concatenated features should be computed;
                set to `None` to collect concatenated features for all
                featurized attributes
        Returns:
           a mapping from each attribute that is featurized to a list containing,
           for each message, the concatenated features (i.e. one matrix per
           per type and sparseness property)
        """
        if used_featurizers is not None:
            assert set(used_featurizers) <= set(
                featurizer.name for featurizer in self.featurizers
            )
        attributes = attributes or set(
            attribute for fd in self.featurizers for attribute in fd.attributes
        )
        collected = {
            attribute: self._rand_featurizer.create_concatenated_features(
                messages,
                attribute=attribute,
                used_featurizers=used_featurizers,
            )
            for attribute in attributes
        }
        return collected
