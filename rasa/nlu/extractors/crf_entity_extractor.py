import logging
import os
import typing

import numpy as np
from typing import Any, Dict, List, Optional, Text, Tuple, NamedTuple, Type

import rasa.nlu.utils.bilou_utils as bilou_utils
import rasa.utils.common as common_utils
from nlu.test import determine_token_labels
from rasa.nlu.tokenizers.spacy_tokenizer import POS_TAG_KEY
from rasa.nlu.config import RasaNLUModelConfig
from rasa.nlu.tokenizers.tokenizer import Tokenizer
from rasa.nlu.components import Component
from rasa.nlu.extractors.extractor import EntityExtractor
from rasa.nlu.model import Metadata
from rasa.nlu.tokenizers.tokenizer import Token
from rasa.nlu.training_data import Message, TrainingData
from rasa.nlu.constants import (
    TOKENS_NAMES,
    TEXT,
    DENSE_FEATURE_NAMES,
    ENTITIES,
    NO_ENTITY_TAG,
    BILOU_ENTITIES,
    ENTITY_ATTRIBUTE_TYPE,
    ENTITY_ATTRIBUTE_GROUP,
    ENTITY_ATTRIBUTE_ROLE,
)
from rasa.constants import DOCS_URL_TRAINING_DATA_NLU, DOCS_URL_COMPONENTS

logger = logging.getLogger(__name__)

if typing.TYPE_CHECKING:
    from sklearn_crfsuite import CRF


ENTITY_LABELS = [ENTITY_ATTRIBUTE_TYPE, ENTITY_ATTRIBUTE_GROUP, ENTITY_ATTRIBUTE_ROLE]


class CRFToken(NamedTuple):
    text: Text
    pos_tag: Text
    pattern: Dict[Text, Any]
    dense_features: np.ndarray
    entity_label: Text
    entity_role_label: Text
    entity_group_label: Text


class CRFEntityExtractor(EntityExtractor):
    @classmethod
    def required_components(cls) -> List[Type[Component]]:
        return [Tokenizer]

    defaults = {
        # BILOU_flag determines whether to use BILOU tagging or not.
        # More rigorous however requires more examples per entity
        # rule of thumb: use only if more than 100 egs. per entity
        "BILOU_flag": True,
        # crf_features is [before, token, after] array with before, token,
        # after holding keys about which features to use for each token,
        # for example, 'title' in array before will have the feature
        # "is the preceding token in title case?"
        # POS features require SpacyTokenizer
        # pattern feature require RegexFeaturizer
        "features": [
            ["low", "title", "upper"],
            [
                "low",
                "bias",
                "prefix5",
                "prefix2",
                "suffix5",
                "suffix3",
                "suffix2",
                "upper",
                "title",
                "digit",
                "pattern",
            ],
            ["low", "title", "upper"],
        ],
        # The maximum number of iterations for optimization algorithms.
        "max_iterations": 50,
        # weight of the L1 regularization
        "L1_c": 0.1,
        # weight of the L2 regularization
        "L2_c": 0.1,
    }

    function_dict = {
        "low": lambda crf_token: crf_token.text.lower(),
        "title": lambda crf_token: crf_token.text.istitle(),
        "prefix5": lambda crf_token: crf_token.text[:5],
        "prefix2": lambda crf_token: crf_token.text[:2],
        "suffix5": lambda crf_token: crf_token.text[-5:],
        "suffix3": lambda crf_token: crf_token.text[-3:],
        "suffix2": lambda crf_token: crf_token.text[-2:],
        "suffix1": lambda crf_token: crf_token.text[-1:],
        "bias": lambda crf_token: "bias",
        "pos": lambda crf_token: crf_token.tag,
        "pos2": lambda crf_token: crf_token.tag[:2]
        if crf_token.tag is not None
        else None,
        "upper": lambda crf_token: crf_token.text.isupper(),
        "digit": lambda crf_token: crf_token.text.isdigit(),
        "pattern": lambda crf_token: crf_token.pattern,
        "text_dense_features": lambda crf_token: crf_token.dense_features,
    }

    def __init__(
        self,
        component_config: Optional[Dict[Text, Any]] = None,
        entity_taggers: Optional[Dict[Text, "CRF"]] = None,
    ) -> None:

        super().__init__(component_config)

        self.entity_taggers = entity_taggers

        self._validate_configuration()

    def _validate_configuration(self) -> None:
        if len(self.component_config.get("features", [])) % 2 != 1:
            raise ValueError(
                "Need an odd number of crf feature lists to have a center word."
            )

    @classmethod
    def required_packages(cls) -> List[Text]:
        return ["sklearn_crfsuite", "sklearn"]

    def train(
        self,
        training_data: TrainingData,
        config: Optional[RasaNLUModelConfig] = None,
        **kwargs: Any,
    ) -> None:
        # checks whether there is at least one
        # example with an entity annotation
        if not training_data.entity_examples:
            logger.debug(
                "No training examples with entities present. Skip training"
                "of 'CRFEntityExtractor'."
            )
            return

        if self.component_config["BILOU_flag"]:
            bilou_utils.apply_bilou_schema(training_data, include_cls_token=False)

        # filter out pre-trained entity examples
        entity_examples = self.filter_trainable_entities(
            training_data.training_examples
        )

        # convert the dataset into features
        # this will train on ALL examples, even the ones without annotations
        dataset = self._create_dataset(entity_examples)

        self._train_model(dataset)

    def _create_dataset(self, examples: List[Message]) -> List[List[CRFToken]]:
        dataset = []

        for example in examples:
            if example.get("entities"):
                # check correct annotation during training
                self._check_correct_annotation(example)

            dataset.append(self._convert_to_crf_tokens(example))

        return dataset

    def process(self, message: Message, **kwargs: Any) -> None:
        entities = self.add_extractor_name(self.extract_entities(message))
        entities = self.clean_up_entities(message, entities)
        message.set(ENTITIES, message.get(ENTITIES, []) + entities, add_to_output=True)

    def extract_entities(self, message: Message) -> List[Dict[Text, Any]]:
        """Take a sentence and return entities in json format"""
        if self.entity_taggers is None:
            return []

        dataset = self._create_dataset([message])
        features = self._sentence_to_features(dataset[0])
        predictions = self.entity_taggers[
            ENTITY_ATTRIBUTE_TYPE
        ].predict_marginals_single(features)
        return self._from_crf_to_json(message, predictions)

    def most_likely_entity(
        self, token_idx: int, entities: List[Dict[Text, float]]
    ) -> Tuple[Text, float]:
        """Get the entity label with the highest confidence.

        Args:
            token_idx: the token index
            entities: list of entity labels and confidence values for a number of tokens

        Returns:
            The entity label and confidence value.

        """
        if len(entities) > token_idx:
            entity_probs = entities[token_idx]
        else:
            entity_probs = None

        if entity_probs is None:
            return "", 0.0

        label = max(entity_probs, key=lambda key: entity_probs[key])

        if self.component_config["BILOU_flag"]:
            # if we are using bilou flags, we will combine the prob
            # of the B, I, L and U tags for an entity (so if we have a
            # score of 60% for `B-address` and 40% and 30%
            # for `I-address`, we will return 70%)
            return (
                label,
                sum([v for k, v in entity_probs.items() if k[2:] == label[2:]]),
            )
        else:
            return label, entity_probs[label]

    @staticmethod
    def _tokens_without_cls(message: Message) -> List[Token]:
        # [:-1] to remove the CLS token from the list of tokens
        return message.get(TOKENS_NAMES[TEXT])[:-1]

    def _from_crf_to_json(
        self, message: Message, predictions: List[Dict[Text, float]]
    ) -> List[Dict[Text, Any]]:

        tokens = self._tokens_without_cls(message)

        if len(tokens) != len(predictions):
            raise Exception(
                "Inconsistency in amount of tokens between crfsuite and message"
            )

        tag_confidence_list = [
            self.most_likely_entity(idx, predictions) for idx in range(len(predictions))
        ]
        confidences = {ENTITY_ATTRIBUTE_TYPE: [x[1] for x in tag_confidence_list]}
        tags = [x[0] for x in tag_confidence_list]

        if self.component_config["BILOU_flag"]:
            tags = bilou_utils.remove_bilou_prefixes(tags)

        tags = {ENTITY_ATTRIBUTE_TYPE: tags}

        return self._convert_tags_to_entities(message.text, tokens, tags, confidences)

    @classmethod
    def load(
        cls,
        meta: Dict[Text, Any],
        model_dir: Text = None,
        model_metadata: Metadata = None,
        cached_component: Optional["CRFEntityExtractor"] = None,
        **kwargs: Any,
    ) -> "CRFEntityExtractor":
        from sklearn.externals import joblib

        file_names = meta.get("file")
        entity_taggers = {}

        for name, file_name in file_names.items():
            model_file = os.path.join(model_dir, file_name)
            if os.path.exists(model_file):
                entity_taggers[name] = joblib.load(model_file)

        return cls(meta, entity_taggers)

    def persist(self, file_name: Text, model_dir: Text) -> Optional[Dict[Text, Any]]:
        """Persist this model into the passed directory.

        Returns the metadata necessary to load the model again."""

        from sklearn.externals import joblib

        file_names = {}

        if self.entity_taggers:
            for name, entity_tagger in self.entity_taggers.items():
                file_name = f"{file_name}.{name}.pkl"
                model_file_name = os.path.join(model_dir, file_name)
                joblib.dump(entity_tagger, model_file_name)
                file_names[name] = file_name

        return {"file": file_names}

    def _sentence_to_features(
        self, sentence: List[CRFToken], tag_features: Optional[List[Text]] = None
    ) -> List[Dict[Text, Any]]:
        """Convert a word into discrete features in self.crf_features,
        including word before and word after."""

        configured_features = self.component_config["features"]
        sentence_features = []

        for word_idx in range(len(sentence)):
            # word before(-1), current word(0), next word(+1)
            feature_span = len(configured_features)
            half_span = feature_span // 2
            feature_range = range(-half_span, half_span + 1)
            prefixes = [str(i) for i in feature_range]
            word_features = {}

            for f_i in feature_range:
                if word_idx + f_i >= len(sentence):
                    word_features["EOS"] = True
                    # End Of Sentence
                elif word_idx + f_i < 0:
                    word_features["BOS"] = True
                    # Beginning Of Sentence
                else:
                    word = sentence[word_idx + f_i]
                    f_i_from_zero = f_i + half_span
                    prefix = prefixes[f_i_from_zero]
                    features = configured_features[f_i_from_zero]
                    for feature in features:
                        if feature == "pattern":
                            # add all regexes as a feature
                            regex_patterns = self.function_dict[feature](word)
                            # pytype: disable=attribute-error
                            for p_name, matched in regex_patterns.items():
                                feature_name = prefix + ":" + feature + ":" + p_name
                                word_features[feature_name] = matched
                            # pytype: enable=attribute-error
                        elif word and (feature == "pos" or feature == "pos2"):
                            value = self.function_dict[feature](word)
                            word_features[f"{prefix}:{feature}"] = value
                        else:
                            # append each feature to a feature vector
                            value = self.function_dict[feature](word)
                            word_features[prefix + ":" + feature] = value

                    if tag_features:
                        word_features[f"{prefix}:entity"] = tag_features[word_idx + f_i]

            sentence_features.append(word_features)
        return sentence_features

    @staticmethod
    def _sentence_to_labels(sentence: List[CRFToken], label_name: Text) -> List[Text]:
        if label_name == ENTITY_ATTRIBUTE_ROLE:
            return [crf_token.entity_role_label for crf_token in sentence]
        if label_name == ENTITY_ATTRIBUTE_GROUP:
            return [crf_token.entity_group_label for crf_token in sentence]

        return [crf_token.entity_label for crf_token in sentence]

    @staticmethod
    def _check_correct_annotation(message: Message) -> None:
        entities = bilou_utils.map_message_entities(message)

        # collect badly annotated examples
        collected = []
        for token, entitiy in zip(message.get(TOKENS_NAMES[TEXT]), entities):
            if entitiy == "-":
                collected.append(token)
            elif collected:
                collected_text = " ".join([t.text for t in collected])
                common_utils.raise_warning(
                    f"Misaligned entity annotation for '{collected_text}' "
                    f"in sentence '{message.text}' with intent "
                    f"'{message.get('intent')}'. "
                    f"Make sure the start and end values of the "
                    f"annotated training examples end at token "
                    f"boundaries (e.g. don't include trailing "
                    f"whitespaces or punctuation).",
                    docs=DOCS_URL_TRAINING_DATA_NLU,
                )
                collected = []

    @staticmethod
    def __pattern_of_token(message: Message, i: int) -> Dict:
        if message.get(TOKENS_NAMES[TEXT]) is not None:
            return message.get(TOKENS_NAMES[TEXT])[i].get("pattern", {})
        else:
            return {}

    @staticmethod
    def __get_dense_features(message: Message) -> Optional[List[Any]]:
        features = message.get(DENSE_FEATURE_NAMES[TEXT])

        if features is None:
            return None

        tokens = message.get(TOKENS_NAMES[TEXT], [])
        if len(tokens) != len(features):
            common_utils.raise_warning(
                f"Number of features ({len(features)}) for attribute "
                f"'{DENSE_FEATURE_NAMES[TEXT]}' "
                f"does not match number of tokens ({len(tokens)}). Set "
                f"'return_sequence' to true in the corresponding featurizer in order "
                f"to make use of the features in 'CRFEntityExtractor'.",
                docs=DOCS_URL_COMPONENTS + "#crfentityextractor",
            )
            return None

        # convert to python-crfsuite feature format
        features_out = []
        for feature in features:
            feature_dict = {
                str(index): token_features
                for index, token_features in enumerate(feature)
            }
            converted = {"text_dense_features": feature_dict}
            features_out.append(converted)
        return features_out

    def _convert_to_crf_tokens(self, message: Message) -> List[CRFToken]:
        """Takes a sentence and switches it to crfsuite format."""
        from rasa.nlu.test import determine_token_labels

        crf_format = []
        tokens = self._tokens_without_cls(message)

        text_dense_features = self.__get_dense_features(message)
        entity_labels = self._get_entity_labels(message)

        for i, token in enumerate(tokens):
            pattern = self.__pattern_of_token(message, i)
            entity = entity_labels[ENTITY_ATTRIBUTE_TYPE][i]
            group = entity_labels[ENTITY_ATTRIBUTE_GROUP][i]
            role = entity_labels[ENTITY_ATTRIBUTE_ROLE][i]
            tag = token.get(POS_TAG_KEY)
            dense_features = (
                text_dense_features[i] if text_dense_features is not None else []
            )

            crf_format.append(
                CRFToken(
                    text=token.text,
                    pos_tag=tag,
                    entity_label=entity,
                    entity_group_label=group,
                    entity_role_label=role,
                    pattern=pattern,
                    dense_features=dense_features,
                )
            )

        return crf_format

    def _get_entity_labels(self, message: Message) -> Dict[Text, List[Text]]:
        tokens = self._tokens_without_cls(message)
        entity_labels = {}

        for name in ENTITY_LABELS:
            if name == ENTITY_ATTRIBUTE_TYPE and self.component_config["BILOU_flag"]:
                # If BILOU tagging is enabled we use the BILOU format for the
                # entity labels
                if message.get(BILOU_ENTITIES):
                    labels = message.get(BILOU_ENTITIES)
                else:
                    labels = [NO_ENTITY_TAG for _ in tokens]
            else:
                labels = [
                    determine_token_labels(
                        token, message.get(ENTITIES), attribute_key=name
                    )
                    for token in tokens
                ]
            entity_labels[name] = labels

        return entity_labels

    def _train_model(self, df_train: List[List[CRFToken]]) -> None:
        """Train the crf tagger based on the training data."""
        import sklearn_crfsuite

        self.entity_taggers = {}

        for name in ENTITY_LABELS:
            X_train = [self._sentence_to_features(sentence) for sentence in df_train]
            y_train = [
                self._sentence_to_labels(sentence, name) for sentence in df_train
            ]

            entity_tagger = sklearn_crfsuite.CRF(
                algorithm="lbfgs",
                # coefficient for L1 penalty
                c1=self.component_config["L1_c"],
                # coefficient for L2 penalty
                c2=self.component_config["L2_c"],
                # stop earlier
                max_iterations=self.component_config["max_iterations"],
                # include transitions that are possible, but not observed
                all_possible_transitions=True,
            )
            entity_tagger.fit(X_train, y_train)

            self.entity_taggers[name] = entity_tagger
