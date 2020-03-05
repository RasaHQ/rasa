import logging
import os
import typing
import numpy as np
from typing import Any, Dict, List, Optional, Text, Tuple, Union, NamedTuple, Type

import rasa.nlu.utils.bilou_utils as bilou_utils
import rasa.utils.common as common_utils
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
)
from rasa.constants import (
    DOCS_URL_TRAINING_DATA_NLU,
    DOCS_URL_COMPONENTS,
    DOCS_URL_MIGRATION_GUIDE,
)

logger = logging.getLogger(__name__)

if typing.TYPE_CHECKING:
    from sklearn_crfsuite import CRF


class CRFToken(NamedTuple):
    text: Text
    tag: Text
    entity: Text
    pattern: Dict[Text, Any]
    dense_features: np.ndarray


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
        ent_tagger: Optional["CRF"] = None,
    ) -> None:

        super().__init__(component_config)

        self.ent_tagger = ent_tagger

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
        if training_data.entity_examples:
            # filter out pre-trained entity examples
            filtered_entity_examples = self.filter_trainable_entities(
                training_data.training_examples
            )

            # convert the dataset into features
            # this will train on ALL examples, even the ones
            # without annotations
            dataset = self._create_dataset(filtered_entity_examples)

            self._train_model(dataset)

    def _create_dataset(self, examples: List[Message]) -> List[List[CRFToken]]:
        dataset = []

        for example in examples:
            entity_offsets = bilou_utils.map_message_entities(example)
            dataset.append(self._from_json_to_crf(example, entity_offsets))

        return dataset

    def process(self, message: Message, **kwargs: Any) -> None:
        extracted = self.add_extractor_name(self.extract_entities(message))
        message.set(ENTITIES, message.get(ENTITIES, []) + extracted, add_to_output=True)

    def extract_entities(self, message: Message) -> List[Dict[Text, Any]]:
        """Take a sentence and return entities in json format"""

        if self.ent_tagger is not None:
            text_data = self._from_text_to_crf(message)
            features = self._sentence_to_features(text_data)
            ents = self.ent_tagger.predict_marginals_single(features)
            return self._from_crf_to_json(message, ents)
        else:
            return []

    def most_likely_entity(self, idx: int, entities: List[Any]) -> Tuple[Text, Any]:
        if len(entities) > idx:
            entity_probs = entities[idx]
        else:
            entity_probs = None
        if entity_probs:
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
        else:
            return "", 0.0

    @staticmethod
    def _create_entity_dict(
        message: Message,
        tokens: List[Token],
        start: int,
        end: int,
        entity: str,
        confidence: float,
    ) -> Dict[Text, Any]:

        _start = tokens[start].start
        _end = tokens[end].end
        value = tokens[start].text
        value += "".join(
            [
                message.text[tokens[i - 1].end : tokens[i].start] + tokens[i].text
                for i in range(start + 1, end + 1)
            ]
        )

        return {
            "start": _start,
            "end": _end,
            "value": value,
            "entity": entity,
            "confidence": confidence,
        }

    @staticmethod
    def _tokens_without_cls(message: Message) -> List[Token]:
        # [:-1] to remove the CLS token from the list of tokens
        return message.get(TOKENS_NAMES[TEXT])[:-1]

    def _find_bilou_end(self, word_idx, entities) -> Any:
        ent_word_idx = word_idx + 1
        finished = False

        # get information about the first word, tagged with `B-...`
        label, confidence = self.most_likely_entity(word_idx, entities)
        entity_label = bilou_utils.entity_name_from_tag(label)

        while not finished:
            label, label_confidence = self.most_likely_entity(ent_word_idx, entities)

            confidence = min(confidence, label_confidence)

            if label[2:] != entity_label:
                # words are not tagged the same entity class
                logger.debug(
                    "Inconsistent BILOU tagging found, B- tag, L- "
                    "tag pair encloses multiple entity classes.i.e. "
                    "[B-a, I-b, L-a] instead of [B-a, I-a, L-a].\n"
                    "Assuming B- class is correct."
                )

            if label.startswith("L-"):
                # end of the entity
                finished = True
            elif label.startswith("I-"):
                # middle part of the entity
                ent_word_idx += 1
            else:
                # entity not closed by an L- tag
                finished = True
                ent_word_idx -= 1
                logger.debug(
                    "Inconsistent BILOU tagging found, B- tag not "
                    "closed by L- tag, i.e [B-a, I-a, O] instead of "
                    "[B-a, L-a, O].\nAssuming last tag is L-"
                )
        return ent_word_idx, confidence

    def _handle_bilou_label(
        self, word_idx: int, entities: List[Any]
    ) -> Tuple[Any, Any, Any]:
        label, confidence = self.most_likely_entity(word_idx, entities)
        entity_label = bilou_utils.entity_name_from_tag(label)

        if bilou_utils.bilou_prefix_from_tag(label) == "U":
            return word_idx, confidence, entity_label

        elif bilou_utils.bilou_prefix_from_tag(label) == "B":
            # start of multi word-entity need to represent whole extent
            ent_word_idx, confidence = self._find_bilou_end(word_idx, entities)
            return ent_word_idx, confidence, entity_label

        else:
            return None, None, None

    def _from_crf_to_json(
        self, message: Message, entities: List[Any]
    ) -> List[Dict[Text, Any]]:

        tokens = self._tokens_without_cls(message)

        if len(tokens) != len(entities):
            raise Exception(
                "Inconsistency in amount of tokens between crfsuite and message"
            )

        if self.component_config["BILOU_flag"]:
            return self._convert_bilou_tagging_to_entity_result(
                message, tokens, entities
            )
        else:
            # not using BILOU tagging scheme, multi-word entities are split.
            return self._convert_simple_tagging_to_entity_result(tokens, entities)

    def _convert_bilou_tagging_to_entity_result(
        self, message: Message, tokens: List[Token], entities: List[Dict[Text, float]]
    ):
        # using the BILOU tagging scheme
        json_ents = []
        word_idx = 0
        while word_idx < len(tokens):
            end_idx, confidence, entity_label = self._handle_bilou_label(
                word_idx, entities
            )

            if end_idx is not None:
                ent = self._create_entity_dict(
                    message, tokens, word_idx, end_idx, entity_label, confidence
                )
                json_ents.append(ent)
                word_idx = end_idx + 1
            else:
                word_idx += 1
        return json_ents

    def _convert_simple_tagging_to_entity_result(
        self, tokens: List[Union[Token, Any]], entities: List[Any]
    ) -> List[Dict[Text, Any]]:
        json_ents = []

        for word_idx in range(len(tokens)):
            entity_label, confidence = self.most_likely_entity(word_idx, entities)
            word = tokens[word_idx]
            if entity_label != NO_ENTITY_TAG:
                ent = {
                    "start": word.start,
                    "end": word.end,
                    "value": word.text,
                    "entity": entity_label,
                    "confidence": confidence,
                }
                json_ents.append(ent)

        return json_ents

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

        file_name = meta.get("file")
        model_file = os.path.join(model_dir, file_name)

        if os.path.exists(model_file):
            ent_tagger = joblib.load(model_file)
            return cls(meta, ent_tagger)
        else:
            return cls(meta)

    def persist(self, file_name: Text, model_dir: Text) -> Optional[Dict[Text, Any]]:
        """Persist this model into the passed directory.

        Returns the metadata necessary to load the model again."""

        from sklearn.externals import joblib

        file_name = file_name + ".pkl"
        if self.ent_tagger:
            model_file_name = os.path.join(model_dir, file_name)
            joblib.dump(self.ent_tagger, model_file_name)

        return {"file": file_name}

    def _sentence_to_features(self, sentence: List[CRFToken]) -> List[Dict[Text, Any]]:
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

            sentence_features.append(word_features)
        return sentence_features

    @staticmethod
    def _sentence_to_labels(
        sentence: List[
            Tuple[
                Optional[Text],
                Optional[Text],
                Text,
                Dict[Text, Any],
                Optional[Dict[str, Any]],
            ]
        ],
    ) -> List[Text]:

        return [label for _, _, label, _, _ in sentence]

    def _from_json_to_crf(
        self, message: Message, entity_offsets: List[Tuple[int, int, Text]]
    ) -> List[CRFToken]:
        """Convert json examples to format of underlying crfsuite."""

        tokens = self._tokens_without_cls(message)
        ents = bilou_utils.bilou_tags_from_offsets(tokens, entity_offsets)

        # collect badly annotated examples
        collected = []
        for t, e in zip(tokens, ents):
            if e == "-":
                collected.append(t)
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

        if not self.component_config["BILOU_flag"]:
            for i, label in enumerate(ents):
                if bilou_utils.bilou_prefix_from_tag(label) in {"B", "I", "U", "L"}:
                    # removes BILOU prefix from label
                    ents[i] = bilou_utils.entity_name_from_tag(label)

        return self._from_text_to_crf(message, ents)

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

    def _from_text_to_crf(
        self, message: Message, entities: List[Text] = None
    ) -> List[CRFToken]:
        """Takes a sentence and switches it to crfsuite format."""

        crf_format = []
        tokens = self._tokens_without_cls(message)

        text_dense_features = self.__get_dense_features(message)

        for i, token in enumerate(tokens):
            pattern = self.__pattern_of_token(message, i)
            entity = entities[i] if entities else "N/A"
            tag = token.get(POS_TAG_KEY)
            dense_features = (
                text_dense_features[i] if text_dense_features is not None else []
            )

            crf_format.append(
                CRFToken(token.text, tag, entity, pattern, dense_features)
            )

        return crf_format

    def _train_model(self, df_train: List[List[CRFToken]]) -> None:
        """Train the crf tagger based on the training data."""
        import sklearn_crfsuite

        X_train = [self._sentence_to_features(sent) for sent in df_train]
        y_train = [self._sentence_to_labels(sent) for sent in df_train]
        self.ent_tagger = sklearn_crfsuite.CRF(
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
        self.ent_tagger.fit(X_train, y_train)
