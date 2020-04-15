import logging
import os
import typing

import numpy as np
from typing import Any, Dict, List, Optional, Text, Tuple, Type

import rasa.nlu.utils.bilou_utils as bilou_utils
import rasa.utils.common as common_utils
from rasa.nlu.test import determine_token_labels
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
    ENTITY_ATTRIBUTE_TYPE,
    ENTITY_ATTRIBUTE_GROUP,
    ENTITY_ATTRIBUTE_ROLE,
)
from rasa.constants import DOCS_URL_TRAINING_DATA_NLU, DOCS_URL_COMPONENTS

logger = logging.getLogger(__name__)

if typing.TYPE_CHECKING:
    from sklearn_crfsuite import CRF


class CRFToken:
    def __init__(
        self,
        text: Text,
        pos_tag: Text,
        pattern: Dict[Text, Any],
        dense_features: np.ndarray,
        entity_tag: Text,
        entity_role_tag: Text,
        entity_group_tag: Text,
    ):
        self.text = text
        self.pos_tag = pos_tag
        self.pattern = pattern
        self.dense_features = dense_features
        self.entity_tag = entity_tag
        self.entity_role_tag = entity_role_tag
        self.entity_group_tag = entity_group_tag


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
        "pos": lambda crf_token: crf_token.pos_tag,
        "pos2": lambda crf_token: crf_token.pos_tag[:2]
        if crf_token.pos_tag is not None
        else None,
        "upper": lambda crf_token: crf_token.text.isupper(),
        "digit": lambda crf_token: crf_token.text.isdigit(),
        "pattern": lambda crf_token: crf_token.pattern,
        "text_dense_features": lambda crf_token: crf_token.dense_features,
        "entity": lambda crf_token: crf_token.entity_tag,
    }

    def __init__(
        self,
        component_config: Optional[Dict[Text, Any]] = None,
        entity_taggers: Optional[Dict[Text, "CRF"]] = None,
    ) -> None:

        super().__init__(component_config)

        self.entity_taggers = entity_taggers

        self.crf_order = [
            ENTITY_ATTRIBUTE_TYPE,
            ENTITY_ATTRIBUTE_ROLE,
            ENTITY_ATTRIBUTE_GROUP,
        ]

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

        self._update_crf_order(training_data)

        # filter out pre-trained entity examples
        entity_examples = self.filter_trainable_entities(
            training_data.training_examples
        )
        dataset = self._create_dataset(entity_examples)

        self._train_model(dataset)

    def _update_crf_order(self, training_data: TrainingData):
        """Train only CRFs we actually have training data for."""
        _crf_order = []

        for tag_name in self.crf_order:
            if tag_name == ENTITY_ATTRIBUTE_TYPE and training_data.entities:
                _crf_order.append(ENTITY_ATTRIBUTE_TYPE)
            elif tag_name == ENTITY_ATTRIBUTE_ROLE and training_data.entity_roles:
                _crf_order.append(ENTITY_ATTRIBUTE_ROLE)
            elif tag_name == ENTITY_ATTRIBUTE_GROUP and training_data.entity_groups:
                _crf_order.append(ENTITY_ATTRIBUTE_GROUP)

        self.crf_order = _crf_order

    def _create_dataset(self, examples: List[Message]) -> List[List[CRFToken]]:
        dataset = []

        for example in examples:
            if example.get(ENTITIES):
                # check correct annotation during training
                self._check_correct_annotation(example)

            dataset.append(self._convert_to_crf_tokens(example))

        return dataset

    def process(self, message: Message, **kwargs: Any) -> None:
        entities = self.extract_entities(message)
        entities = self.add_extractor_name(entities)
        entities = self.clean_up_entities(message, entities)
        message.set(ENTITIES, message.get(ENTITIES, []) + entities, add_to_output=True)

    def extract_entities(self, message: Message) -> List[Dict[Text, Any]]:
        """Take a sentence and return entities in json format"""
        if self.entity_taggers is None:
            return []

        crf_tokens = self._create_dataset([message])[0]

        predictions = {}
        for name, entity_tagger in self.entity_taggers.items():
            include_tag_features = name != ENTITY_ATTRIBUTE_TYPE

            if include_tag_features:
                self._add_tag_to_crf_token(crf_tokens, predictions)

            features = self._sentence_to_features(crf_tokens, include_tag_features)
            predictions[name] = entity_tagger.predict_marginals_single(features)

        return self._create_entities(message, predictions)

    def _add_tag_to_crf_token(
        self,
        crf_tokens: List[CRFToken],
        predictions: Dict[Text, List[Dict[Text, float]]],
    ):
        """Add predicted entity tags to CRF tokens."""

        if ENTITY_ATTRIBUTE_TYPE in predictions:
            tag_confidence_list = [
                self._most_likely_entity(prediction)
                for prediction in predictions[ENTITY_ATTRIBUTE_TYPE]
            ]
            for tag_conf, token in zip(tag_confidence_list, crf_tokens):
                token.entity_tag = tag_conf[0]

    def _most_likely_entity(
        self, entity_predictions: Optional[Dict[Text, float]]
    ) -> Tuple[Text, float]:
        """Get the entity tag with the highest confidence.

        Args:
            entity_predictions: mapping of entity tag to confidence value

        Returns:
            The entity tag and confidence value.
        """
        if entity_predictions is None:
            return "", 0.0

        tag = max(entity_predictions, key=lambda key: entity_predictions[key])

        if self.component_config["BILOU_flag"]:
            # if we are using BILOU flags, we will combine the prob
            # of the B, I, L and U tags for an entity (so if we have a
            # score of 60% for `B-address` and 40% and 30%
            # for `I-address`, we will return 70%)
            return (
                tag,
                sum(
                    [
                        _confidence
                        for _tag, _confidence in entity_predictions.items()
                        if bilou_utils.tag_without_prefix(tag)
                        == bilou_utils.tag_without_prefix(_tag)
                    ]
                ),
            )
        else:
            return tag, entity_predictions[tag]

    @staticmethod
    def _tokens_without_cls(message: Message) -> List[Token]:
        # [:-1] to remove the CLS token from the list of tokens
        return message.get(TOKENS_NAMES[TEXT])[:-1]

    def _create_entities(
        self, message: Message, predictions: Dict[Text, List[Dict[Text, float]]]
    ) -> List[Dict[Text, Any]]:
        tokens = self._tokens_without_cls(message)

        tags = {}
        confidences = {}

        for name, predicted_tags in predictions.items():
            if len(tokens) != len(predicted_tags):
                raise Exception(
                    "Inconsistency in amount of tokens between crfsuite and message"
                )

            tag_confidence_list = [
                self._most_likely_entity(prediction) for prediction in predicted_tags
            ]
            confidences[name] = [x[1] for x in tag_confidence_list]
            _tags = [x[0] for x in tag_confidence_list]

            if self.component_config["BILOU_flag"]:
                _tags = bilou_utils.remove_bilou_prefixes(_tags)

            tags[name] = _tags

        return self.convert_tags_to_entities(message.text, tokens, tags, confidences)

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
        self, sentence: List[CRFToken], include_tag_features: bool = False
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
                    # End Of Sentence
                    word_features["EOS"] = True
                elif word_idx + f_i < 0:
                    # Beginning Of Sentence
                    word_features["BOS"] = True
                else:
                    word = sentence[word_idx + f_i]
                    f_i_from_zero = f_i + half_span
                    prefix = prefixes[f_i_from_zero]

                    features = configured_features[f_i_from_zero]
                    if include_tag_features:
                        features.append("entity")

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

            sentence_features.append(word_features)

        return sentence_features

    @staticmethod
    def _sentence_to_tags(sentence: List[CRFToken], tag_name: Text) -> List[Text]:
        if tag_name == ENTITY_ATTRIBUTE_ROLE:
            return [crf_token.entity_role_tag for crf_token in sentence]
        if tag_name == ENTITY_ATTRIBUTE_GROUP:
            return [crf_token.entity_group_tag for crf_token in sentence]

        return [crf_token.entity_tag for crf_token in sentence]

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

        crf_format = []
        tokens = self._tokens_without_cls(message)

        text_dense_features = self.__get_dense_features(message)
        tags = self._get_tags(message)

        for i, token in enumerate(tokens):
            pattern = self.__pattern_of_token(message, i)
            entity = self.__get_tag_for(tags, i, ENTITY_ATTRIBUTE_TYPE)
            group = self.__get_tag_for(tags, i, ENTITY_ATTRIBUTE_GROUP)
            role = self.__get_tag_for(tags, i, ENTITY_ATTRIBUTE_ROLE)
            pos_tag = token.get(POS_TAG_KEY)
            dense_features = (
                text_dense_features[i] if text_dense_features is not None else []
            )

            crf_format.append(
                CRFToken(
                    text=token.text,
                    pos_tag=pos_tag,
                    entity_tag=entity,
                    entity_group_tag=group,
                    entity_role_tag=role,
                    pattern=pattern,
                    dense_features=dense_features,
                )
            )

        return crf_format

    @staticmethod
    def __get_tag_for(
        tags: Dict[Text, List[Text]], token_idx: int, tag_name: Text
    ) -> Optional[Text]:
        return tags[tag_name][token_idx] if tag_name in tags else None

    def _get_tags(self, message: Message) -> Dict[Text, List[Text]]:
        tokens = self._tokens_without_cls(message)
        tags = {}

        for tag_name in self.crf_order:
            if self.component_config["BILOU_flag"]:
                bilou_key = bilou_utils.get_bilou_key_for_tag(tag_name)
                if message.get(bilou_key):
                    _tags = message.get(bilou_key)
                else:
                    _tags = [NO_ENTITY_TAG for _ in tokens]
            else:
                _tags = [
                    determine_token_labels(
                        token, message.get(ENTITIES), attribute_key=tag_name
                    )
                    for token in tokens
                ]
            tags[tag_name] = _tags

        return tags

    def _train_model(self, df_train: List[List[CRFToken]]) -> None:
        """Train the crf tagger based on the training data."""
        import sklearn_crfsuite

        self.entity_taggers = {}

        for tag_name in self.crf_order:
            include_tag_features = tag_name != ENTITY_ATTRIBUTE_TYPE
            X_train = [
                self._sentence_to_features(sentence, include_tag_features)
                for sentence in df_train
            ]
            y_train = [
                self._sentence_to_tags(sentence, tag_name) for sentence in df_train
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

            self.entity_taggers[tag_name] = entity_tagger
