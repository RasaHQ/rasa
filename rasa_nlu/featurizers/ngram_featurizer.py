from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import typing
from builtins import map
from builtins import range
import logging
import os
import re
import time
import warnings
import io
from collections import Counter
from string import punctuation
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from future.utils import PY3
from typing import Text

from rasa_nlu.components import Component
from rasa_nlu.config import RasaNLUConfig
from rasa_nlu.featurizers import Featurizer
from rasa_nlu.training_data import Message
from rasa_nlu.training_data import TrainingData

logger = logging.getLogger(__name__)

if typing.TYPE_CHECKING:
    from spacy.language import Language
    import numpy as np
    from rasa_nlu.model import Metadata


class NGramFeaturizer(Featurizer):
    name = "intent_featurizer_ngrams"

    provides = ["text_features"]

    requires = ["spacy_doc"]

    n_gram_min_length = 3

    n_gram_max_length = 17

    n_gram_min_occurrences = 5

    min_intent_examples_for_ngram_classification = 10

    def __init__(self):
        self.best_num_ngrams = None
        self.all_ngrams = None

    @classmethod
    def required_packages(cls):
        # type: () -> List[Text]
        return ["spacy", "numpy", "sklearn", "cloudpickle"]

    def train(self, training_data, config, **kwargs):
        # type: (TrainingData, RasaNLUConfig, **Any) -> None

        start = time.time()
        self.train_on_sentences(training_data.intent_examples, config["max_number_of_ngrams"])
        logger.debug("Ngram collection took {} seconds".format(time.time() - start))

        for example in training_data.training_examples:
            updated = self._text_features_with_ngrams(example, self.best_num_ngrams)
            example.set("text_features", updated)

    def process(self, message, **kwargs):
        # type: (Message, **Any) -> None

        updated = self._text_features_with_ngrams(message, self.best_num_ngrams)
        message.set("text_features", updated)

    def _text_features_with_ngrams(self, message, max_ngrams):
        import numpy as np

        ngrams_to_use = self._ngrams_to_use(max_ngrams)

        if ngrams_to_use is not None:
            extras = np.array(self._ngrams_in_sentence(message, ngrams_to_use))
            return self._combine_with_existing_text_features(message, extras)
        else:
            return message.get("text_features")

    @classmethod
    def load(cls, model_dir=None, model_metadata=None, cached_component=None, **kwargs):
        # type: (Text, Metadata, Optional[Component], **Any) -> NGramFeaturizer
        import cloudpickle

        if model_dir and model_metadata.get("ngram_featurizer"):
            classifier_file = os.path.join(model_dir, model_metadata.get("ngram_featurizer"))
            with io.open(classifier_file, 'rb') as f:   # pramga: no cover
                if PY3:
                    return cloudpickle.load(f, encoding="latin-1")
                else:
                    return cloudpickle.load(f)
        else:
            return NGramFeaturizer()

    def persist(self, model_dir):
        # type: (Text) -> Dict[Text, Any]
        """Persist this model into the passed directory. Returns the metadata necessary to load the model again."""
        import cloudpickle

        classifier_file = os.path.join(model_dir, "ngram_featurizer.pkl")
        with io.open(classifier_file, 'wb') as f:
            cloudpickle.dump(self, f)

        return {
            "ngram_featurizer": "ngram_featurizer.pkl"
        }

    def train_on_sentences(self, examples, max_number_of_ngrams):
        labels = [e.get("intent") for e in examples]
        self.all_ngrams = self._get_best_ngrams(examples, labels)
        self.best_num_ngrams = self._cross_validation(examples, labels, max_number_of_ngrams)

    def _ngrams_to_use(self, num_ngrams):
        if num_ngrams == 0 or self.all_ngrams is None:
            return []
        elif num_ngrams is not None:
            return self.all_ngrams[:num_ngrams]
        else:
            return self.all_ngrams

    def _get_best_ngrams(self, examples, labels):
        """Returns an ordered list of the best character ngrams for an intent classification problem"""

        oov_strings = self._remove_in_vocab_words(examples)
        ngrams = self._generate_all_ngrams(oov_strings)
        return self._sort_applicable_ngrams(ngrams, examples, labels)

    def _remove_in_vocab_words(self, examples):
        """Automatically removes words with digits in them, that may be a
        hyperlink or that _are_ in vocabulary for the nlp"""

        new_sents = []
        for example in examples:
            new_sents.append(self._remove_in_vocab_words_from_sentence(example))
        return new_sents

    def _remove_in_vocab_words_from_sentence(self, example):
        """Automatically removes words with digits in them, hyperlink and in-vocab-words."""

        cleaned_tokens = []
        for token in example.get("spacy_doc"):
            if not token.has_vector and not token.like_url and \
                    not token.like_num and not token.like_email and not token.is_punct:
                cleaned_tokens.append(token)

        # keep only out-of-vocab 'non_word' words
        non_words = ' '.join([t.text for t in cleaned_tokens])

        # remove digits and extra spaces
        non_words = ''.join([letter for letter in non_words if not letter.isdigit()])
        non_words = ' '.join([word for word in non_words.split(' ') if word != ''])

        # add cleaned sentence to list of these sentences
        return non_words

    def _sort_applicable_ngrams(self, list_of_ngrams, examples, labels):
        """Given an intent classification problem and a list of ngrams, creates ordered list of most useful ngrams."""

        if list_of_ngrams:
            from sklearn import linear_model, preprocessing
            import numpy as np

            # filter examples where we do not have enough labeled instances for cv
            usable_labels = []
            for label in np.unique(labels):
                lab_sents = np.array(examples)[np.array(labels) == label]
                if len(lab_sents) < self.min_intent_examples_for_ngram_classification:
                    continue
                usable_labels.append(label)

            mask = [label in usable_labels for label in labels]
            if any(mask) and len(usable_labels) >= 2:
                try:
                    examples = np.array(examples)[mask]
                    labels = np.array(labels)[mask]

                    X = np.array(self._ngrams_in_sentences(examples, list_of_ngrams))
                    intent_encoder = preprocessing.LabelEncoder()
                    intent_encoder.fit(labels)
                    y = intent_encoder.transform(labels)

                    clf = linear_model.RandomizedLogisticRegression(C=1)
                    clf.fit(X, y)
                    scores = clf.scores_
                    sort_idx = [i[0] for i in sorted(enumerate(scores), key=lambda x: -1 * x[1])]

                    return np.array(list_of_ngrams)[sort_idx]
                except ValueError as e:
                    if "needs samples of at least 2 classes" in str(e):
                        # we got unlucky during the random sampling :( and selected a slice that only contains one class
                        return []
                    else:
                        raise e
            else:
                # there is no example we can use for the cross validation
                return []
        else:
            return []

    def _ngrams_in_sentences(self, examples, ngrams):
        """Given a set of sentences, returns a feature vector for each sentence.

        The first $k$ elements are from the `intent_features`,
        the rest are {1,0} elements denoting whether an ngram is in sentence."""

        all_vectors = []
        for example in examples:
            presence_vector = self._ngrams_in_sentence(example, ngrams)
            all_vectors.append(presence_vector)
        return all_vectors

    def _ngrams_in_sentence(self, example, ngrams):
        """Given a set of sentences, returns a vector of {1,0} values indicating ngram presence"""

        import numpy as np

        cleaned_sentence = self._remove_in_vocab_words_from_sentence(example)
        presence_vector = np.zeros(len(ngrams))
        idx_array = [idx for idx in range(len(ngrams)) if ngrams[idx] in cleaned_sentence]
        presence_vector[idx_array] = 1
        return presence_vector

    def _generate_all_ngrams(self, list_of_strings):
        """Takes a list of strings and generates all character ngrams.

        Generated ngrams are at least 3 characters (and at most 17), occur at least 5 times
        and occur independently of longer superset ngrams at least once."""

        features = {}
        counters = {self.n_gram_min_length - 1: Counter()}

        for n in range(self.n_gram_min_length, self.n_gram_max_length):
            candidates = []
            features[n] = []
            counters[n] = Counter()

            # generate all possible n length ngrams
            for text in list_of_strings:
                text = text.replace(punctuation, ' ')
                for word in text.lower().split(' '):
                    cands = [word[i:i + n] for i in range(len(word) - n)]
                    for cand in cands:
                        counters[n][cand] += 1
                        if cand not in candidates:
                            candidates.append(cand)

            # iterate over these candidates picking only the applicable ones
            for can in candidates:
                if counters[n][can] >= self.n_gram_min_occurrences:
                    features[n].append(can)
                    begin = can[:-1]
                    end = can[1:]
                    if n >= self.n_gram_min_length:
                        if counters[n - 1][begin] == counters[n][can] and begin in features[n - 1]:
                            features[n - 1].remove(begin)
                        if counters[n - 1][end] == counters[n][can] and end in features[n - 1]:
                            features[n - 1].remove(end)

        return [item for sublist in list(features.values()) for item in sublist]

    def _cross_validation(self, examples, labels, max_ngrams):
        """choose the best number of ngrams to include in bow.

        Given an intent classification problem and a set of ordered ngrams (ordered in terms
        of importance by pick_applicable_ngrams) we choose the best number of ngrams to include
        in our bow vecs by cross validation."""

        from sklearn import preprocessing
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import cross_val_score
        import numpy as np

        if examples:
            collected_features = [e.get("text_features") for e in examples if e.get("text_features") is not None]
        else:
            collected_features = []

        existing_text_features = np.stack(collected_features) if collected_features else None

        def features_with_ngrams(max_ngrams):
            ngrams_to_use = self._ngrams_to_use(max_ngrams)
            extras = np.array(self._ngrams_in_sentences(examples, ngrams_to_use))
            if existing_text_features is not None:
                return np.hstack((existing_text_features, extras))
            else:
                return extras

        clf2 = LogisticRegression(class_weight='balanced')
        intent_encoder = preprocessing.LabelEncoder()
        intent_encoder.fit(labels)
        y = intent_encoder.transform(labels)
        cv_splits = min(10, np.min(np.bincount(y))) if y.size > 0 else 0
        if cv_splits >= 3:
            logger.debug("Started ngram cross-validation to find best number of ngrams to use...")
            num_ngrams = np.unique(list(map(int, np.floor(np.linspace(1, max_ngrams, 8)))))
            if existing_text_features is not None:
                no_ngrams_X = features_with_ngrams(max_ngrams=0)
                no_ngrams_score = np.mean(cross_val_score(clf2, no_ngrams_X, y, cv=cv_splits))
            else:
                no_ngrams_score = 0.0
            scores = []
            for n in num_ngrams:
                X = features_with_ngrams(max_ngrams=n)
                score = np.mean(cross_val_score(clf2, X, y, cv=cv_splits))
                scores.append(score)
                logger.debug("Evaluating usage of {} ngrams. Score: {}".format(n, score))
            n_top = num_ngrams[np.argmax(scores)]
            logger.debug("Score without ngrams: {}".format(no_ngrams_score))
            logger.info("Best score with {} ngrams: {}".format(n_top, np.max(scores)))
            return n_top
        else:
            warnings.warn("Can't cross-validate ngram featurizer. There aren't enough examples per intent (at least 3)")
            return max_ngrams
