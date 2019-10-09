import logging
import os
import re
from typing import Any, Dict, List, Optional, Text, Union
import numpy as np

from sklearn.feature_extraction.text import CountVectorizer
from rasa.nlu import utils
from rasa.nlu.config import RasaNLUModelConfig
from rasa.nlu.featurizers import Featurizer
from rasa.nlu.model import Metadata
from rasa.nlu.training_data import Message, TrainingData

logger = logging.getLogger(__name__)

from rasa.nlu.constants import (
    MESSAGE_RESPONSE_ATTRIBUTE,
    MESSAGE_INTENT_ATTRIBUTE,
    MESSAGE_TEXT_ATTRIBUTE,
    MESSAGE_TOKENS_NAMES,
    MESSAGE_ATTRIBUTES,
    MESSAGE_SPACY_FEATURES_NAMES,
    MESSAGE_VECTOR_FEATURE_NAMES,
    SPACY_FEATURIZABLE_ATTRIBUTES,
)


class CountVectorsFeaturizer(Featurizer):
    """Bag of words featurizer

    Creates bag-of-words representation of intent features
    using sklearn's `CountVectorizer`.
    All tokens which consist only of digits (e.g. 123 and 99
    but not ab12d) will be represented by a single feature.

    Set `analyzer` to 'char_wb'
    to use the idea of Subword Semantic Hashing
    from https://arxiv.org/abs/1810.07150.
    """

    provides = [
        MESSAGE_VECTOR_FEATURE_NAMES[attribute] for attribute in MESSAGE_ATTRIBUTES
    ]

    requires = []

    defaults = {
        # whether to use a shared vocab
        "use_shared_vocab": False,
        # the parameters are taken from
        # sklearn's CountVectorizer
        # whether to use word or character n-grams
        # 'char_wb' creates character n-grams inside word boundaries
        # n-grams at the edges of words are padded with space.
        "analyzer": "word",  # use 'char' or 'char_wb' for character
        # regular expression for tokens
        # only used if analyzer == 'word'
        "token_pattern": r"(?u)\b\w\w+\b",
        # remove accents during the preprocessing step
        "strip_accents": None,  # {'ascii', 'unicode', None}
        # list of stop words
        "stop_words": None,  # string {'english'}, list, or None (default)
        # min document frequency of a word to add to vocabulary
        # float - the parameter represents a proportion of documents
        # integer - absolute counts
        "min_df": 1,  # float in range [0.0, 1.0] or int
        # max document frequency of a word to add to vocabulary
        # float - the parameter represents a proportion of documents
        # integer - absolute counts
        "max_df": 1.0,  # float in range [0.0, 1.0] or int
        # set range of ngrams to be extracted
        "min_ngram": 1,  # int
        "max_ngram": 1,  # int
        # limit vocabulary size
        "max_features": None,  # int or None
        # if convert all characters to lowercase
        "lowercase": True,  # bool
        # handling Out-Of-Vacabulary (OOV) words
        # will be converted to lowercase if lowercase is True
        "OOV_token": None,  # string or None
        "OOV_words": [],  # string or list of strings
    }

    @classmethod
    def required_packages(cls) -> List[Text]:
        return ["sklearn"]

    def _load_count_vect_params(self):

        # Use shared vocabulary between text and all other attributes of Message
        self.use_shared_vocab = self.component_config["use_shared_vocab"]

        # set analyzer
        self.analyzer = self.component_config["analyzer"]

        # regular expression for tokens
        self.token_pattern = self.component_config["token_pattern"]

        # remove accents during the preprocessing step
        self.strip_accents = self.component_config["strip_accents"]

        # list of stop words
        self.stop_words = self.component_config["stop_words"]

        # min number of word occurancies in the document to add to vocabulary
        self.min_df = self.component_config["min_df"]

        # max number (fraction if float) of word occurancies
        # in the document to add to vocabulary
        self.max_df = self.component_config["max_df"]

        # set ngram range
        self.min_ngram = self.component_config["min_ngram"]
        self.max_ngram = self.component_config["max_ngram"]

        # limit vocabulary size
        self.max_features = self.component_config["max_features"]

        # if convert all characters to lowercase
        self.lowercase = self.component_config["lowercase"]

    # noinspection PyPep8Naming
    def _load_OOV_params(self):
        self.OOV_token = self.component_config["OOV_token"]

        self.OOV_words = self.component_config["OOV_words"]
        if self.OOV_words and not self.OOV_token:
            logger.error(
                "The list OOV_words={} was given, but "
                "OOV_token was not. OOV words are ignored."
                "".format(self.OOV_words)
            )
            self.OOV_words = []

        if self.lowercase and self.OOV_token:
            # convert to lowercase
            self.OOV_token = self.OOV_token.lower()
            if self.OOV_words:
                self.OOV_words = [w.lower() for w in self.OOV_words]

    def _check_attribute_vocabulary(self, attribute: Text) -> bool:
        """Check if trained vocabulary exists in attribute's count vectorizer"""
        try:
            return hasattr(self.vectorizers[attribute], "vocabulary_")
        except (AttributeError, TypeError):
            return False

    def _get_attribute_vocabulary(self, attribute: Text) -> Optional[Dict[Text, int]]:
        """Get trained vocabulary from attribute's count vectorizer"""

        try:
            return self.vectorizers[attribute].vocabulary_
        except (AttributeError, TypeError):
            return None

    def _collect_vectorizer_vocabularies(self):
        """Get vocabulary for all attributes"""

        attribute_vocabularies = {}
        for attribute in MESSAGE_ATTRIBUTES:
            attribute_vocabularies[attribute] = self._get_attribute_vocabulary(
                attribute
            )
        return attribute_vocabularies

    def _get_attribute_vocabulary_tokens(self, attribute: Text) -> Optional[List[Text]]:
        """Get all keys of vocabulary of an attribute"""

        attribute_vocabulary = self._get_attribute_vocabulary(attribute)
        try:
            return list(attribute_vocabulary.keys())
        except TypeError:
            return None

    def _check_analyzer(self):
        if self.analyzer != "word":
            if self.OOV_token is not None:
                logger.warning(
                    "Analyzer is set to character, "
                    "provided OOV word token will be ignored."
                )
            if self.stop_words is not None:
                logger.warning(
                    "Analyzer is set to character, "
                    "provided stop words will be ignored."
                )
            if self.max_ngram == 1:
                logger.warning(
                    "Analyzer is set to character, "
                    "but max n-gram is set to 1. "
                    "It means that the vocabulary will "
                    "contain single letters only."
                )

    def __init__(
        self,
        component_config: Dict[Text, Any] = None,
        vectorizers: Optional[Dict[Text, "CountVectorizer"]] = None,
    ) -> None:
        """Construct a new count vectorizer using the sklearn framework."""

        super(CountVectorsFeaturizer, self).__init__(component_config)

        # parameters for sklearn's CountVectorizer
        self._load_count_vect_params()

        # handling Out-Of-Vocabulary (OOV) words
        self._load_OOV_params()

        # warn that some of config parameters might be ignored
        self._check_analyzer()

        # declare class instance for CountVectorizer
        self.vectorizers = vectorizers

    def _get_message_text_by_attribute(
        self, message: "Message", attribute: Text = MESSAGE_TEXT_ATTRIBUTE
    ) -> Text:
        """Get processed text of attribute of a message"""

        if message.get(attribute) is None:
            # return empty string since sklearn countvectorizer does not like None object while training and predicting
            return ""

        tokens = self._get_message_tokens_by_attribute(message, attribute)

        text = self._process_text(tokens, attribute)

        text = self._replace_with_oov_token(text, attribute)

        return text

    def _process_text(
        self, tokens: List[Text], attribute: Text = MESSAGE_TEXT_ATTRIBUTE
    ) -> Text:
        """Apply processing and cleaning steps to text"""

        text = " ".join(tokens)

        if attribute == MESSAGE_INTENT_ATTRIBUTE:
            # Don't do any processing for intent attribute. Treat them as whole labels
            return text

        # replace all digits with NUMBER token
        text = re.sub(r"\b[0-9]+\b", "__NUMBER__", text)

        # convert to lowercase if necessary
        if self.lowercase:
            text = text.lower()
        return text

    def _replace_with_oov_token(self, text: Text, attribute: Text) -> Text:
        """Replace OOV words with OOV token"""

        if self.OOV_token and self.analyzer == "word":
            text_tokens = text.split()
            if self._check_attribute_vocabulary(
                attribute
            ) and self.OOV_token in self._get_attribute_vocabulary(attribute):
                # CountVectorizer is trained, process for prediction
                text_tokens = [
                    t
                    if t in self._get_attribute_vocabulary_tokens(attribute)
                    else self.OOV_token
                    for t in text_tokens
                ]
            elif self.OOV_words:
                # CountVectorizer is not trained, process for train
                text_tokens = [
                    self.OOV_token if t in self.OOV_words else t for t in text_tokens
                ]
            text = " ".join(text_tokens)
        return text

    @staticmethod
    def _get_message_tokens_by_attribute(
        message: "Message", attribute: Text
    ) -> List[Text]:
        """Get text tokens of an attribute of a message"""

        if attribute in SPACY_FEATURIZABLE_ATTRIBUTES and message.get(
            MESSAGE_SPACY_FEATURES_NAMES[attribute]
        ):  # if lemmatize is possible
            tokens = [
                t.lemma_ for t in message.get(MESSAGE_SPACY_FEATURES_NAMES[attribute])
            ]
        elif message.get(
            MESSAGE_TOKENS_NAMES[attribute]
        ):  # if directly tokens is provided
            tokens = [t.text for t in message.get(MESSAGE_TOKENS_NAMES[attribute])]
        else:
            tokens = message.get(attribute).split()
        return tokens

    # noinspection PyPep8Naming
    def _check_OOV_present(self, examples):
        """Check if an OOV word is present"""
        if self.OOV_token and not self.OOV_words:
            for t in examples:
                if (
                    t is None
                    or self.OOV_token in t
                    or (self.lowercase and self.OOV_token in t.lower())
                ):
                    return

            logger.warning(
                "OOV_token='{}' was given, but it is not present "
                "in the training data. All unseen words "
                "will be ignored during prediction."
                "".format(self.OOV_token)
            )

    def _set_attribute_features(
        self,
        attribute: Text,
        attribute_features: np.ndarray,
        training_data: "TrainingData",
    ):
        """Set computed features of the attribute to corresponding message objects"""
        for i, example in enumerate(training_data.intent_examples):
            # create bag for each example
            example.set(
                MESSAGE_VECTOR_FEATURE_NAMES[attribute],
                self._combine_with_existing_features(
                    example,
                    attribute_features[i],
                    MESSAGE_VECTOR_FEATURE_NAMES[attribute],
                ),
            )

    def _get_all_attributes_processed_texts(
        self, training_data: "TrainingData"
    ) -> Dict[Text, List[Text]]:
        """Get processed text for all attributes of examples in training data"""

        processed_attribute_texts = {}
        for attribute in MESSAGE_ATTRIBUTES:
            attribute_texts = [
                self._get_message_text_by_attribute(example, attribute)
                for example in training_data.intent_examples
            ]
            self._check_OOV_present(attribute_texts)
            processed_attribute_texts[attribute] = attribute_texts
        return processed_attribute_texts

    @staticmethod
    def create_shared_vocab_vectorizers(
        token_pattern,
        strip_accents,
        lowercase,
        stop_words,
        ngram_range,
        max_df,
        min_df,
        max_features,
        analyzer,
        vocabulary=None,
    ) -> Dict[Text, "CountVectorizer"]:
        """Create vectorizers for all attributes with shared vocabulary"""

        shared_vectorizer = CountVectorizer(
            token_pattern=token_pattern,
            strip_accents=strip_accents,
            lowercase=lowercase,
            stop_words=stop_words,
            ngram_range=ngram_range,
            max_df=max_df,
            min_df=min_df,
            max_features=max_features,
            analyzer=analyzer,
            vocabulary=vocabulary,
        )

        attribute_vectorizers = {}

        for attribute in MESSAGE_ATTRIBUTES:
            attribute_vectorizers[attribute] = shared_vectorizer

        return attribute_vectorizers

    @staticmethod
    def create_independent_vocab_vectorizers(
        token_pattern,
        strip_accents,
        lowercase,
        stop_words,
        ngram_range,
        max_df,
        min_df,
        max_features,
        analyzer,
        vocabulary=None,
    ) -> Dict[Text, "CountVectorizer"]:
        """Create vectorizers for all attributes with independent vocabulary"""

        attribute_vectorizers = {}

        for attribute in MESSAGE_ATTRIBUTES:

            attribute_vocabulary = vocabulary[attribute] if vocabulary else None

            attribute_vectorizer = CountVectorizer(
                token_pattern=token_pattern,
                strip_accents=strip_accents,
                lowercase=lowercase,
                stop_words=stop_words,
                ngram_range=ngram_range,
                max_df=max_df,
                min_df=min_df,
                max_features=max_features,
                analyzer=analyzer,
                vocabulary=attribute_vocabulary,
            )
            attribute_vectorizers[attribute] = attribute_vectorizer

        return attribute_vectorizers

    def _train_with_shared_vocab(self, attribute_texts: Dict[Text, List[Text]]):
        """Construct the vectorizers and train them with a shared vocab"""

        self.vectorizers = self.create_shared_vocab_vectorizers(
            self.token_pattern,
            self.strip_accents,
            self.lowercase,
            self.stop_words,
            (self.min_ngram, self.max_ngram),
            self.max_df,
            self.min_df,
            self.max_features,
            self.analyzer,
        )

        combined_cleaned_texts = []
        for attribute in MESSAGE_ATTRIBUTES:
            combined_cleaned_texts += attribute_texts[attribute]

        try:
            self.vectorizers[MESSAGE_TEXT_ATTRIBUTE].fit(combined_cleaned_texts)
        except ValueError:
            logger.warning(
                "Unable to train a shared CountVectorizer. Leaving an untrained CountVectorizer"
            )

    @staticmethod
    def _attribute_texts_is_non_empty(attribute_texts):
        return any(attribute_texts)

    def _train_with_independent_vocab(self, attribute_texts: Dict[Text, List[Text]]):
        """Construct the vectorizers and train them with an independent vocab"""

        self.vectorizers = self.create_independent_vocab_vectorizers(
            self.token_pattern,
            self.strip_accents,
            self.lowercase,
            self.stop_words,
            (self.min_ngram, self.max_ngram),
            self.max_df,
            self.min_df,
            self.max_features,
            self.analyzer,
        )

        for attribute in MESSAGE_ATTRIBUTES:
            if self._attribute_texts_is_non_empty(attribute_texts[attribute]):
                try:
                    self.vectorizers[attribute].fit(attribute_texts[attribute])
                except ValueError:
                    logger.warning(
                        "Unable to train CountVectorizer for message attribute {}. "
                        "Leaving an untrained CountVectorizer for it".format(attribute)
                    )
            else:
                logger.debug(
                    "No text provided for {} attribute in any messages of training data. Skipping "
                    "training a CountVectorizer for it.".format(attribute)
                )

    def _get_featurized_attribute(
        self, attribute: Text, attribute_texts: List[Text]
    ) -> Optional[np.ndarray]:
        """Return features of a particular attribute for complete data"""

        if self._check_attribute_vocabulary(attribute):
            # count vectorizer was trained
            featurized_attributes = (
                self.vectorizers[attribute].transform(attribute_texts).toarray()
            )
            return featurized_attributes
        else:
            return None

    def train(
        self, training_data: TrainingData, cfg: RasaNLUModelConfig = None, **kwargs: Any
    ) -> None:
        """Train the featurizer.

        Take parameters from config and
        construct a new count vectorizer using the sklearn framework.
        """

        spacy_nlp = kwargs.get("spacy_nlp")
        if spacy_nlp is not None:
            # create spacy lemma_ for OOV_words
            self.OOV_words = [t.lemma_ for w in self.OOV_words for t in spacy_nlp(w)]

        # process sentences and collect data for all attributes
        processed_attribute_texts = self._get_all_attributes_processed_texts(
            training_data
        )

        # train for all attributes
        if self.use_shared_vocab:
            self._train_with_shared_vocab(processed_attribute_texts)
        else:
            self._train_with_independent_vocab(processed_attribute_texts)

        # transform for all attributes
        for attribute in MESSAGE_ATTRIBUTES:

            attribute_features = self._get_featurized_attribute(
                attribute, processed_attribute_texts[attribute]
            )

            if attribute_features is not None:
                self._set_attribute_features(
                    attribute, attribute_features, training_data
                )

    def process(self, message: Message, **kwargs: Any) -> None:
        """Process incoming message and compute and set features"""

        if self.vectorizers is None:
            logger.error(
                "There is no trained CountVectorizer: "
                "component is either not trained or "
                "didn't receive enough training data"
            )
        else:
            message_text = self._get_message_text_by_attribute(
                message, attribute=MESSAGE_TEXT_ATTRIBUTE
            )

            bag = (
                self.vectorizers[MESSAGE_TEXT_ATTRIBUTE]
                .transform([message_text])
                .toarray()
                .squeeze()
            )
            message.set(
                MESSAGE_VECTOR_FEATURE_NAMES[MESSAGE_TEXT_ATTRIBUTE],
                self._combine_with_existing_features(
                    message,
                    bag,
                    feature_name=MESSAGE_VECTOR_FEATURE_NAMES[MESSAGE_TEXT_ATTRIBUTE],
                ),
            )

    @staticmethod
    def _is_any_model_trained(attribute_vocabularies) -> bool:
        """Check if any model got trained"""

        return any(value is not None for value in attribute_vocabularies.values())

    def persist(self, file_name: Text, model_dir: Text) -> Optional[Dict[Text, Any]]:
        """Persist this model into the passed directory.

        Returns the metadata necessary to load the model again.
        """

        file_name = file_name + ".pkl"

        if self.vectorizers:
            # vectorizer instance was not None, some models could have been trained
            attribute_vocabularies = self._collect_vectorizer_vocabularies()
            if self._is_any_model_trained(attribute_vocabularies):
                # Definitely need to persist some vocabularies
                featurizer_file = os.path.join(model_dir, file_name)

                if self.use_shared_vocab:
                    # Only persist vocabulary from one attribute. Can be loaded and distributed to all attributes.
                    utils.json_pickle(
                        featurizer_file, attribute_vocabularies[MESSAGE_TEXT_ATTRIBUTE]
                    )
                else:
                    utils.json_pickle(featurizer_file, attribute_vocabularies)
        return {"file": file_name}

    @classmethod
    def load(
        cls,
        meta: Dict[Text, Any],
        model_dir: Text = None,
        model_metadata: Metadata = None,
        cached_component: Optional["CountVectorsFeaturizer"] = None,
        **kwargs: Any
    ) -> "CountVectorsFeaturizer":

        file_name = meta.get("file")
        featurizer_file = os.path.join(model_dir, file_name)

        if os.path.exists(featurizer_file):
            vocabulary = utils.json_unpickle(featurizer_file)

            share_vocabulary = meta["use_shared_vocab"]

            if share_vocabulary:
                vectorizers = cls.create_shared_vocab_vectorizers(
                    token_pattern=meta["token_pattern"],
                    strip_accents=meta["strip_accents"],
                    lowercase=meta["lowercase"],
                    stop_words=meta["stop_words"],
                    ngram_range=(meta["min_ngram"], meta["max_ngram"]),
                    max_df=meta["max_df"],
                    min_df=meta["min_df"],
                    max_features=meta["max_features"],
                    analyzer=meta["analyzer"],
                    vocabulary=vocabulary,
                )
            else:
                vectorizers = cls.create_independent_vocab_vectorizers(
                    token_pattern=meta["token_pattern"],
                    strip_accents=meta["strip_accents"],
                    lowercase=meta["lowercase"],
                    stop_words=meta["stop_words"],
                    ngram_range=(meta["min_ngram"], meta["max_ngram"]),
                    max_df=meta["max_df"],
                    min_df=meta["min_df"],
                    max_features=meta["max_features"],
                    analyzer=meta["analyzer"],
                    vocabulary=vocabulary,
                )

            return cls(meta, vectorizers)
        else:
            return cls(meta)
