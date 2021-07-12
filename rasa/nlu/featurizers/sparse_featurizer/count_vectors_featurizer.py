import logging
import os
import re
import scipy.sparse
from typing import Any, Dict, List, Optional, Text, Type, Tuple, Set

import rasa.shared.utils.io
from rasa.shared.constants import DOCS_URL_COMPONENTS
import rasa.utils.io as io_utils
from sklearn.feature_extraction.text import CountVectorizer
from rasa.nlu.config import RasaNLUModelConfig
from rasa.nlu.tokenizers.tokenizer import Tokenizer
from rasa.nlu.components import Component
from rasa.nlu.featurizers.featurizer import SparseFeaturizer
from rasa.shared.nlu.training_data.features import Features
from rasa.nlu.model import Metadata
from rasa.shared.nlu.training_data.training_data import TrainingData
from rasa.shared.nlu.training_data.message import Message
from rasa.shared.exceptions import RasaException
from rasa.nlu.constants import (
    TOKENS_NAMES,
    MESSAGE_ATTRIBUTES,
    DENSE_FEATURIZABLE_ATTRIBUTES,
    FEATURIZER_CLASS_ALIAS,
)
from rasa.shared.nlu.constants import (
    TEXT,
    INTENT,
    INTENT_RESPONSE_KEY,
    RESPONSE,
    ACTION_TEXT,
    FEATURE_TYPE_SENTENCE,
    FEATURE_TYPE_SEQUENCE,
    ACTION_NAME,
)

BUFFER_SLOTS_PREFIX = "buf_"

logger = logging.getLogger(__name__)


class CountVectorsFeaturizer(SparseFeaturizer):
    """Creates a sequence of token counts features based on sklearn's `CountVectorizer`.

    All tokens which consist only of digits (e.g. 123 and 99
    but not ab12d) will be represented by a single feature.

    Set `analyzer` to 'char_wb'
    to use the idea of Subword Semantic Hashing
    from https://arxiv.org/abs/1810.07150.
    """

    @classmethod
    def required_components(cls) -> List[Type[Component]]:
        return [Tokenizer]

    defaults = {
        # whether to use a shared vocab
        "use_shared_vocab": False,
        # the parameters are taken from
        # sklearn's CountVectorizer
        # whether to use word or character n-grams
        # 'char_wb' creates character n-grams inside word boundaries
        # n-grams at the edges of words are padded with space.
        "analyzer": "word",  # use 'char' or 'char_wb' for character
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
        # handling Out-Of-Vocabulary (OOV) words
        # will be converted to lowercase if lowercase is True
        "OOV_token": None,  # string or None
        "OOV_words": [],  # string or list of strings
        # indicates whether the featurizer should use the lemma of a word for
        # counting (if available) or not
        "use_lemma": True,
        # Additional vocabulary size to be kept reserved for finetuning
        "additional_vocabulary_size": {TEXT: None, RESPONSE: None, ACTION_TEXT: None},
    }

    @classmethod
    def required_packages(cls) -> List[Text]:
        return ["sklearn"]

    def _load_count_vect_params(self) -> None:

        # Use shared vocabulary between text and all other attributes of Message
        self.use_shared_vocab = self.component_config["use_shared_vocab"]

        # set analyzer
        self.analyzer = self.component_config["analyzer"]

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

        # use the lemma of the words or not
        self.use_lemma = self.component_config["use_lemma"]

    def _load_vocabulary_params(self) -> None:
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
        additional_size_attributes = [
            key
            for key, value in self.component_config[
                "additional_vocabulary_size"
            ].items()
            if value
        ]
        if additional_size_attributes:
            rasa.shared.utils.io.raise_deprecation_warning(
                f"The parameter `additional_vocabulary_size` has been specified for "
                f"attributes - `{additional_size_attributes}`. The parameter has been "
                f"deprecated since the pipeline no longer creates an extra buffer for "
                f"additional vocabulary. Any value assigned to "
                f"this parameter will be ignored. You can omit specifying "
                f"`additional_vocabulary_size` in future runs."
            )

    def _check_attribute_vocabulary(self, attribute: Text) -> bool:
        """Checks if trained vocabulary exists in attribute's count vectorizer."""
        try:
            return hasattr(self.vectorizers[attribute], "vocabulary_")
        except (AttributeError, KeyError):
            return False

    def _get_attribute_vocabulary(self, attribute: Text) -> Optional[Dict[Text, int]]:
        """Get trained vocabulary from attribute's count vectorizer"""

        try:
            return self.vectorizers[attribute].vocabulary_
        except (AttributeError, TypeError):
            return None

    def _get_attribute_vocabulary_tokens(self, attribute: Text) -> Optional[List[Text]]:
        """Get all keys of vocabulary of an attribute"""

        attribute_vocabulary = self._get_attribute_vocabulary(attribute)
        try:
            return list(attribute_vocabulary.keys())
        except TypeError:
            return None

    def _check_analyzer(self) -> None:
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

    @staticmethod
    def _attributes_for(analyzer: Text) -> List[Text]:
        """Create a list of attributes that should be featurized."""

        # intents should be featurized only by word level count vectorizer
        return (
            MESSAGE_ATTRIBUTES if analyzer == "word" else DENSE_FEATURIZABLE_ATTRIBUTES
        )

    def __init__(
        self,
        component_config: Optional[Dict[Text, Any]] = None,
        vectorizers: Optional[Dict[Text, "CountVectorizer"]] = None,
        finetune_mode: bool = False,
    ) -> None:
        """Construct a new count vectorizer using the sklearn framework."""
        super().__init__(component_config)

        # parameters for sklearn's CountVectorizer
        self._load_count_vect_params()

        # handling Out-Of-Vocabulary (OOV) words
        self._load_vocabulary_params()

        # warn that some of config parameters might be ignored
        self._check_analyzer()

        # set which attributes to featurize
        self._attributes = self._attributes_for(self.analyzer)

        # declare class instance for CountVectorizer
        self.vectorizers = vectorizers or {}

        self.finetune_mode = finetune_mode

    def _get_message_tokens_by_attribute(
        self, message: "Message", attribute: Text
    ) -> List[Text]:
        """Get text tokens of an attribute of a message"""
        if message.get(TOKENS_NAMES[attribute]):
            return [
                t.lemma if self.use_lemma else t.text
                for t in message.get(TOKENS_NAMES[attribute])
            ]
        else:
            return []

    def _process_tokens(self, tokens: List[Text], attribute: Text = TEXT) -> List[Text]:
        """Apply processing and cleaning steps to text"""

        if attribute in [INTENT, ACTION_NAME, INTENT_RESPONSE_KEY]:
            # Don't do any processing for intent attribute. Treat them as whole labels
            return tokens

        # replace all digits with NUMBER token
        tokens = [re.sub(r"\b[0-9]+\b", "__NUMBER__", text) for text in tokens]

        # convert to lowercase if necessary
        if self.lowercase:
            tokens = [text.lower() for text in tokens]

        return tokens

    def _replace_with_oov_token(
        self, tokens: List[Text], attribute: Text
    ) -> List[Text]:
        """Replace OOV words with OOV token"""

        if self.OOV_token and self.analyzer == "word":
            vocabulary_exists = self._check_attribute_vocabulary(attribute)
            if vocabulary_exists and self.OOV_token in self._get_attribute_vocabulary(
                attribute
            ):
                # CountVectorizer is trained, process for prediction
                tokens = [
                    t
                    if t in self._get_attribute_vocabulary_tokens(attribute)
                    else self.OOV_token
                    for t in tokens
                ]
            elif self.OOV_words:
                # CountVectorizer is not trained, process for train
                tokens = [self.OOV_token if t in self.OOV_words else t for t in tokens]

        return tokens

    def _get_processed_message_tokens_by_attribute(
        self, message: Message, attribute: Text = TEXT
    ) -> List[Text]:
        """Get processed text of attribute of a message"""

        if message.get(attribute) is None:
            # return empty list since sklearn countvectorizer does not like None
            # object while training and predicting
            return []

        tokens = self._get_message_tokens_by_attribute(message, attribute)
        tokens = self._process_tokens(tokens, attribute)
        tokens = self._replace_with_oov_token(tokens, attribute)

        return tokens

    # noinspection PyPep8Naming
    def _check_OOV_present(self, all_tokens: List[List[Text]], attribute: Text) -> None:
        """Check if an OOV word is present"""
        if not self.OOV_token or self.OOV_words or not all_tokens:
            return

        for tokens in all_tokens:
            for text in tokens:
                if self.OOV_token in text or (
                    self.lowercase and self.OOV_token in text.lower()
                ):
                    return

        if any(text for tokens in all_tokens for text in tokens):
            training_data_type = "NLU" if attribute == TEXT else "ResponseSelector"

            # if there is some text in tokens, warn if there is no oov token
            rasa.shared.utils.io.raise_warning(
                f"The out of vocabulary token '{self.OOV_token}' was configured, but "
                f"could not be found in any one of the {training_data_type} "
                f"training examples. All unseen words will be "
                f"ignored during prediction.",
                docs=DOCS_URL_COMPONENTS + "#countvectorsfeaturizer",
            )

    def _get_all_attributes_processed_tokens(
        self, training_data: TrainingData
    ) -> Dict[Text, List[List[Text]]]:
        """Get processed text for all attributes of examples in training data"""

        processed_attribute_tokens = {}
        for attribute in self._attributes:
            all_tokens = [
                self._get_processed_message_tokens_by_attribute(example, attribute)
                for example in training_data.training_examples
            ]
            if attribute in DENSE_FEATURIZABLE_ATTRIBUTES:
                # check for oov tokens only in text based attributes
                self._check_OOV_present(all_tokens, attribute)
            processed_attribute_tokens[attribute] = all_tokens

        return processed_attribute_tokens

    @staticmethod
    def _convert_attribute_tokens_to_texts(
        attribute_tokens: Dict[Text, List[List[Text]]]
    ) -> Dict[Text, List[Text]]:
        attribute_texts = {}

        for attribute in attribute_tokens.keys():
            list_of_tokens = attribute_tokens[attribute]
            attribute_texts[attribute] = [" ".join(tokens) for tokens in list_of_tokens]

        return attribute_texts

    def _update_vectorizer_vocabulary(
        self, attribute: Text, new_vocabulary: Set[Text]
    ) -> None:
        """Updates the existing vocabulary of the vectorizer with new unseen words.

        Args:
            attribute: Message attribute for which vocabulary should be updated.
            new_vocabulary: Set of words to expand the vocabulary with if they are
                unseen.
        """
        existing_vocabulary: Dict[Text, int] = self.vectorizers[attribute].vocabulary
        self._merge_new_vocabulary_tokens(existing_vocabulary, new_vocabulary)
        self._set_vocabulary(attribute, existing_vocabulary)

    def _merge_new_vocabulary_tokens(
        self, existing_vocabulary: Dict[Text, int], vocabulary: Set[Text]
    ) -> None:
        """Merges new vocabulary tokens with the existing vocabulary.

        New vocabulary items should always be added to the end of the existing
        vocabulary and the order of the existing vocabulary should not be disturbed.

        Args:
            existing_vocabulary: existing vocabulary
            vocabulary: set of new tokens

        Raises:
            RasaException: if `use_shared_vocab` is set to True and there are new
                           vocabulary items added during incremental training.
        """
        for token in vocabulary:
            if token not in existing_vocabulary:
                if self.use_shared_vocab:
                    raise RasaException(
                        "Using a shared vocabulary in `CountVectorsFeaturizer` is not "
                        "supported during incremental training since it requires "
                        "dynamically adjusting layers that correspond to label "
                        f"attributes such as {INTENT_RESPONSE_KEY}, {INTENT}, etc. "
                        "This is currently not possible. In order to avoid this "
                        "exception we suggest to set `use_shared_vocab=False` or train"
                        " from scratch."
                    )
                existing_vocabulary[token] = len(existing_vocabulary)

    def _set_vocabulary(
        self, attribute: Text, original_vocabulary: Dict[Text, int]
    ) -> None:
        """Sets the vocabulary of the vectorizer of attribute.

        Args:
            attribute: Message attribute for which vocabulary should be set
            original_vocabulary: Vocabulary for the attribute to be set.
        """
        self.vectorizers[attribute].vocabulary_ = original_vocabulary
        self.vectorizers[attribute]._validate_vocabulary()

    @staticmethod
    def _construct_vocabulary_from_texts(
        vectorizer: CountVectorizer, texts: List[Text]
    ) -> Set:
        """Applies vectorizer's preprocessor on texts to get the vocabulary from texts.

        Args:
            vectorizer: Sklearn's count vectorizer which has been pre-configured.
            texts: Examples from which the vocabulary should be constructed

        Returns:
            Unique vocabulary words extracted.
        """
        analyzer = vectorizer.build_analyzer()
        vocabulary_words = set()
        for example in texts:
            example_vocabulary: List[Text] = analyzer(example)
            vocabulary_words.update(example_vocabulary)
        return vocabulary_words

    @staticmethod
    def _attribute_texts_is_non_empty(attribute_texts: List[Text]) -> bool:
        return any(attribute_texts)

    def _train_with_shared_vocab(self, attribute_texts: Dict[Text, List[Text]]) -> None:
        """Constructs the vectorizers and train them with a shared vocab."""
        combined_cleaned_texts = []
        for attribute in self._attributes:
            combined_cleaned_texts += attribute_texts[attribute]

        # To train a shared vocabulary, we use TEXT as the
        # attribute for which a combined vocabulary is built.
        if not self.finetune_mode:
            self.vectorizers = self._create_shared_vocab_vectorizers(
                {
                    "strip_accents": self.strip_accents,
                    "lowercase": self.lowercase,
                    "stop_words": self.stop_words,
                    "min_ngram": self.min_ngram,
                    "max_ngram": self.max_ngram,
                    "max_df": self.max_df,
                    "min_df": self.min_df,
                    "max_features": self.max_features,
                    "analyzer": self.analyzer,
                }
            )
            self._fit_vectorizer_from_scratch(TEXT, combined_cleaned_texts)
        else:
            self._fit_loaded_vectorizer(TEXT, combined_cleaned_texts)
        self._log_vocabulary_stats(TEXT)

    def _train_with_independent_vocab(
        self, attribute_texts: Dict[Text, List[Text]]
    ) -> None:
        """Constructs the vectorizers and train them with an independent vocab."""
        if not self.finetune_mode:
            self.vectorizers = self._create_independent_vocab_vectorizers(
                {
                    "strip_accents": self.strip_accents,
                    "lowercase": self.lowercase,
                    "stop_words": self.stop_words,
                    "min_ngram": self.min_ngram,
                    "max_ngram": self.max_ngram,
                    "max_df": self.max_df,
                    "min_df": self.min_df,
                    "max_features": self.max_features,
                    "analyzer": self.analyzer,
                }
            )
        for attribute in self._attributes:
            if self._attribute_texts_is_non_empty(attribute_texts[attribute]):
                if not self.finetune_mode:
                    self._fit_vectorizer_from_scratch(
                        attribute, attribute_texts[attribute]
                    )
                else:
                    self._fit_loaded_vectorizer(attribute, attribute_texts[attribute])

                self._log_vocabulary_stats(attribute)
            else:
                logger.debug(
                    f"No text provided for {attribute} attribute in any messages of "
                    f"training data. Skipping training a CountVectorizer for it."
                )

    def _log_vocabulary_stats(self, attribute: Text) -> None:
        """Logs number of vocabulary items that were created for a specified attribute.

        Args:
            attribute: Message attribute for which vocabulary stats are logged.
        """
        if attribute in DENSE_FEATURIZABLE_ATTRIBUTES:
            vocabulary_size = len(self.vectorizers[attribute].vocabulary_)
            logger.info(
                f"{vocabulary_size} vocabulary items "
                f"were created for {attribute} attribute."
            )

    def _fit_loaded_vectorizer(
        self, attribute: Text, attribute_texts: List[Text]
    ) -> None:
        """Fits training texts to a previously trained count vectorizer.

        We do not use the `.fit()` method because the new unseen
        words should occupy the buffer slots of the vocabulary.

        Args:
            attribute: Message attribute for which the vectorizer is to be trained.
            attribute_texts: Training texts for the attribute
        """
        # Get vocabulary words by the preprocessor
        new_vocabulary = self._construct_vocabulary_from_texts(
            self.vectorizers[attribute], attribute_texts
        )
        # update the vocabulary of vectorizer with new vocabulary
        self._update_vectorizer_vocabulary(attribute, new_vocabulary)

    def _fit_vectorizer_from_scratch(
        self, attribute: Text, attribute_texts: List[Text]
    ) -> None:
        """Fits training texts to an untrained count vectorizer.

        Args:
            attribute: Message attribute for which the vectorizer is to be trained.
            attribute_texts: Training texts for the attribute
        """
        try:
            self.vectorizers[attribute].fit(attribute_texts)
        except ValueError:
            logger.warning(
                f"Unable to train CountVectorizer for message "
                f"attribute {attribute} since the call to sklearn's "
                f"`.fit()` method failed. Leaving an untrained "
                f"CountVectorizer for it."
            )

    def _create_features(
        self, attribute: Text, all_tokens: List[List[Text]]
    ) -> Tuple[
        List[Optional[scipy.sparse.spmatrix]], List[Optional[scipy.sparse.spmatrix]]
    ]:
        if not self.vectorizers.get(attribute):
            return [None], [None]

        sequence_features = []
        sentence_features = []

        for i, tokens in enumerate(all_tokens):
            if not tokens:
                # nothing to featurize
                sequence_features.append(None)
                sentence_features.append(None)
                continue

            # vectorizer.transform returns a sparse matrix of size
            # [n_samples, n_features]
            # set input to list of tokens if sequence should be returned
            # otherwise join all tokens to a single string and pass that as a list
            if not tokens:
                # attribute is not set (e.g. response not present)
                sequence_features.append(None)
                sentence_features.append(None)
                continue

            seq_vec = self.vectorizers[attribute].transform(tokens)
            seq_vec.sort_indices()

            sequence_features.append(seq_vec.tocoo())

            if attribute in DENSE_FEATURIZABLE_ATTRIBUTES:
                tokens_text = [" ".join(tokens)]
                sentence_vec = self.vectorizers[attribute].transform(tokens_text)
                sentence_vec.sort_indices()

                sentence_features.append(sentence_vec.tocoo())
            else:
                sentence_features.append(None)

        return sequence_features, sentence_features

    def _get_featurized_attribute(
        self, attribute: Text, all_tokens: List[List[Text]]
    ) -> Tuple[
        List[Optional[scipy.sparse.spmatrix]], List[Optional[scipy.sparse.spmatrix]]
    ]:
        """Return features of a particular attribute for complete data"""

        if self._check_attribute_vocabulary(attribute):
            # count vectorizer was trained
            return self._create_features(attribute, all_tokens)
        else:
            return [], []

    def _set_attribute_features(
        self,
        attribute: Text,
        sequence_features: List[scipy.sparse.spmatrix],
        sentence_features: List[scipy.sparse.spmatrix],
        examples: List[Message],
    ) -> None:
        """Set computed features of the attribute to corresponding message objects"""
        for i, message in enumerate(examples):
            # create bag for each example
            if sequence_features[i] is not None:
                final_sequence_features = Features(
                    sequence_features[i],
                    FEATURE_TYPE_SEQUENCE,
                    attribute,
                    self.component_config[FEATURIZER_CLASS_ALIAS],
                )
                message.add_features(final_sequence_features)

            if sentence_features[i] is not None:
                final_sentence_features = Features(
                    sentence_features[i],
                    FEATURE_TYPE_SENTENCE,
                    attribute,
                    self.component_config[FEATURIZER_CLASS_ALIAS],
                )
                message.add_features(final_sentence_features)

    def train(
        self,
        training_data: TrainingData,
        cfg: Optional[RasaNLUModelConfig] = None,
        **kwargs: Any,
    ) -> None:
        """Train the featurizer.

        Take parameters from config and
        construct a new count vectorizer using the sklearn framework.
        """

        spacy_nlp = kwargs.get("spacy_nlp")
        if spacy_nlp is not None:
            # create spacy lemma_ for OOV_words
            self.OOV_words = [
                t.lemma_ if self.use_lemma else t.text
                for w in self.OOV_words
                for t in spacy_nlp(w)
            ]

        # process sentences and collect data for all attributes
        processed_attribute_tokens = self._get_all_attributes_processed_tokens(
            training_data
        )

        # train for all attributes
        attribute_texts = self._convert_attribute_tokens_to_texts(
            processed_attribute_tokens
        )
        if self.use_shared_vocab:
            self._train_with_shared_vocab(attribute_texts)
        else:
            self._train_with_independent_vocab(attribute_texts)

        # transform for all attributes
        for attribute in self._attributes:
            sequence_features, sentence_features = self._get_featurized_attribute(
                attribute, processed_attribute_tokens[attribute]
            )

            if sequence_features and sentence_features:
                self._set_attribute_features(
                    attribute,
                    sequence_features,
                    sentence_features,
                    training_data.training_examples,
                )

    def process(self, message: Message, **kwargs: Any) -> None:
        """Process incoming message and compute and set features"""

        if self.vectorizers is None:
            logger.error(
                "There is no trained CountVectorizer: "
                "component is either not trained or "
                "didn't receive enough training data"
            )
            return
        for attribute in self._attributes:

            message_tokens = self._get_processed_message_tokens_by_attribute(
                message, attribute
            )

            # features shape (1, seq, dim)
            sequence_features, sentence_features = self._create_features(
                attribute, [message_tokens]
            )

            self._set_attribute_features(
                attribute, sequence_features, sentence_features, [message]
            )

    def _collect_vectorizer_vocabularies(self) -> Dict[Text, Optional[Dict[Text, int]]]:
        """Get vocabulary for all attributes"""

        attribute_vocabularies = {}
        for attribute in self._attributes:
            attribute_vocabularies[attribute] = self._get_attribute_vocabulary(
                attribute
            )
        return attribute_vocabularies

    @staticmethod
    def _is_any_model_trained(
        attribute_vocabularies: Dict[Text, Optional[Dict[Text, int]]]
    ) -> bool:
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
                    # Only persist vocabulary from one attribute. Can be loaded and
                    # distributed to all attributes.
                    vocab = attribute_vocabularies[TEXT]
                else:
                    vocab = attribute_vocabularies

                io_utils.json_pickle(featurizer_file, vocab)

        return {"file": file_name}

    @classmethod
    def _create_shared_vocab_vectorizers(
        cls, parameters: Dict[Text, Any], vocabulary: Optional[Any] = None
    ) -> Dict[Text, CountVectorizer]:
        """Create vectorizers for all attributes with shared vocabulary"""

        shared_vectorizer = CountVectorizer(
            token_pattern=r"(?u)\b\w+\b" if parameters["analyzer"] == "word" else None,
            strip_accents=parameters["strip_accents"],
            lowercase=parameters["lowercase"],
            stop_words=parameters["stop_words"],
            ngram_range=(parameters["min_ngram"], parameters["max_ngram"]),
            max_df=parameters["max_df"],
            min_df=parameters["min_df"],
            max_features=parameters["max_features"],
            analyzer=parameters["analyzer"],
            vocabulary=vocabulary,
        )

        attribute_vectorizers = {}

        for attribute in cls._attributes_for(parameters["analyzer"]):
            attribute_vectorizers[attribute] = shared_vectorizer

        return attribute_vectorizers

    @classmethod
    def _create_independent_vocab_vectorizers(
        cls, parameters: Dict[Text, Any], vocabulary: Optional[Any] = None
    ) -> Dict[Text, CountVectorizer]:
        """Create vectorizers for all attributes with independent vocabulary"""

        attribute_vectorizers = {}

        for attribute in cls._attributes_for(parameters["analyzer"]):

            attribute_vocabulary = vocabulary[attribute] if vocabulary else None

            attribute_vectorizer = CountVectorizer(
                token_pattern=r"(?u)\b\w+\b"
                if parameters["analyzer"] == "word"
                else None,
                strip_accents=parameters["strip_accents"],
                lowercase=parameters["lowercase"],
                stop_words=parameters["stop_words"],
                ngram_range=(parameters["min_ngram"], parameters["max_ngram"]),
                max_df=parameters["max_df"],
                min_df=parameters["min_df"],
                max_features=parameters["max_features"],
                analyzer=parameters["analyzer"],
                vocabulary=attribute_vocabulary,
            )
            attribute_vectorizers[attribute] = attribute_vectorizer

        return attribute_vectorizers

    @classmethod
    def load(
        cls,
        meta: Dict[Text, Any],
        model_dir: Text,
        model_metadata: Optional[Metadata] = None,
        cached_component: Optional["CountVectorsFeaturizer"] = None,
        should_finetune: bool = False,
        **kwargs: Any,
    ) -> "CountVectorsFeaturizer":
        """Loads trained component (see parent class for full docstring)."""
        file_name = meta.get("file")
        featurizer_file = os.path.join(model_dir, file_name)

        if not os.path.exists(featurizer_file):
            return cls(meta)

        vocabulary = io_utils.json_unpickle(featurizer_file)

        share_vocabulary = meta["use_shared_vocab"]

        if share_vocabulary:
            vectorizers = cls._create_shared_vocab_vectorizers(
                meta, vocabulary=vocabulary
            )
        else:
            vectorizers = cls._create_independent_vocab_vectorizers(
                meta, vocabulary=vocabulary
            )

        ftr = cls(meta, vectorizers, should_finetune)

        # make sure the vocabulary has been loaded correctly
        for attribute in vectorizers:
            ftr.vectorizers[attribute]._validate_vocabulary()

        return ftr
