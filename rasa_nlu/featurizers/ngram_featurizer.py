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
from rasa_nlu.training_data import TrainingData


if typing.TYPE_CHECKING:
    from spacy.language import Language
    import numpy as np


class NGramFeaturizer(Component):
    name = "intent_featurizer_ngrams"

    context_provides = {
        "train": ["intent_features"],
        "process": ["intent_features"],
    }

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

    def train(self, training_data, intent_features, spacy_nlp, max_number_of_ngrams):
        # type: (TrainingData, List[float], Language, Optional[int]) -> Dict[Text, Any]

        start = time.time()
        labels = [e['intent'] for e in training_data.intent_examples]
        sentences = [e['text'] for e in training_data.intent_examples]
        self.all_ngrams = self._get_best_ngrams(sentences, labels, spacy_nlp)
        self.best_num_ngrams = self._cross_validation(
            sentences, labels, intent_features, spacy_nlp, max_number_of_ngrams)
        logging.debug("Ngram collection took {} seconds".format(time.time() - start))
        stacked = self._create_bow_vecs(intent_features, sentences, spacy_nlp, max_ngrams=self.best_num_ngrams)
        return {"intent_features": stacked}

    def process(self, intent_features, text, spacy_nlp):
        # type: (List[float], Text, Language) -> Dict[Text, Any]
        import numpy as np

        if self.all_ngrams is not None:
            ngrams_to_use = self._ngrams_to_use(self.best_num_ngrams)
            if ngrams_to_use is None:
                return {"intent_features": intent_features}

            extras = np.array(self._ngrams_in_sentence(text, spacy_nlp, ngrams_to_use))
            total = np.hstack((intent_features, extras))
            return {"intent_features": total}
        else:
            return {"intent_features": intent_features}

    @classmethod
    def load(cls, model_dir, ngram_featurizer):
        # type: (Text, Text) -> NGramFeaturizer
        import cloudpickle

        if model_dir and ngram_featurizer:
            classifier_file = os.path.join(model_dir, ngram_featurizer)
            with io.open(classifier_file, 'rb') as f:
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

    def _ngrams_to_use(self, num_ngrams):
        if num_ngrams == 0:
            return None
        elif num_ngrams is not None:
            return self.all_ngrams[:num_ngrams]
        else:
            return self.all_ngrams

    def _get_best_ngrams(self, sentences, labels, spacy_nlp):
        """Returns an ordered list of the best character ngrams for an intent classification problem"""

        oov_strings = self._remove_in_vocab_words(sentences, spacy_nlp)
        ngrams = self._generate_all_ngrams(oov_strings)
        return self._sort_applicable_ngrams(ngrams, sentences, labels, spacy_nlp)

    def _remove_in_vocab_words(self, sentences, spacy_nlp):
        """Automatically removes words with digits in them, that may be a
        hyperlink or that _are_ in vocabulary for the nlp"""

        new_sents = []
        for sentence in sentences:
            new_sents.append(self._remove_in_vocab_words_from_sentence(sentence, spacy_nlp))
        return new_sents

    def _remove_hyperlinks(self, sentence):
        return re.sub(r'^https?:\/\/.*[\r\n]*', '', sentence, flags=re.MULTILINE)

    def _remove_punctuation(self, sentence):
        return ''.join([letter for letter in sentence if letter not in punctuation])

    def _remove_in_vocab_words_from_sentence(self, sentence, spacy_nlp):
        """Automatically removes words with digits in them, hyperlink and in-vocab-words."""

        cleaned_sentence = self._remove_hyperlinks(self._remove_punctuation(sentence))

        # keep only out-of-vocab 'non_word' words
        doc = spacy_nlp(cleaned_sentence)
        non_words = ' '.join([q.lower_ for q in doc if not q.has_vector])

        # remove digits and extra spaces
        non_words = ''.join([letter for letter in non_words if not letter.isdigit()])
        non_words = ' '.join([word for word in non_words.split(' ') if word != ''])

        # add cleaned sentence to list of these sentences
        return non_words

    def _sort_applicable_ngrams(self, list_of_ngrams, sentences, labels, spacy_nlp):
        """Given an intent classification problem and a list of ngrams, creates ordered list of most useful ngrams."""

        if list_of_ngrams:
            from sklearn import linear_model, preprocessing
            import numpy as np

            usable_labels = []
            for label in np.unique(labels):
                lab_sents = np.array(sentences)[np.array(labels) == label]
                if len(lab_sents) < self.min_intent_examples_for_ngram_classification:
                    continue
                usable_labels.append(label)

            mask = [label in usable_labels for label in labels]
            sentences = np.array(sentences)[mask]
            labels = np.array(labels)[mask]

            X = np.array(self._ngrams_in_sentences(sentences, spacy_nlp, list_of_ngrams))
            intent_encoder = preprocessing.LabelEncoder()
            intent_encoder.fit(labels)
            y = intent_encoder.transform(labels)

            clf = linear_model.RandomizedLogisticRegression(C=1)
            clf.fit(X, y)
            scores = clf.scores_
            sort_idx = [i[0] for i in sorted(enumerate(scores), key=lambda x: -1 * x[1])]

            return np.array(list_of_ngrams)[sort_idx]
        else:
            return []

    def _ngrams_in_sentences(self, sentences, spacy_nlp, ngrams):
        """Given a set of sentences, returns a vector of {1,0} values indicating ngram presence"""

        all_vectors = []
        for sentence in sentences:
            presence_vector = self._ngrams_in_sentence(sentence, spacy_nlp, ngrams)
            all_vectors.append(presence_vector)
        return all_vectors

    def _ngrams_in_sentence(self, sentence, spacy_nlp, ngrams):
        """Given a set of sentences, returns a vector of {1,0} values indicating ngram presence"""

        import numpy as np

        cleaned_sentence = self._remove_in_vocab_words_from_sentence(sentence, spacy_nlp)
        presence_vector = np.zeros(len(ngrams))
        idx_array = [idx for idx in range(len(ngrams)) if ngrams[idx] in cleaned_sentence]
        presence_vector[idx_array] = 1
        return presence_vector

    def _generate_all_ngrams(self, list_of_strings):
        """Takes a list of strings and generates all character ngrams.

        Generated ngrams are at least 3 characters (and at most 17), occur at least 5 times
        and occur independently of longer superset ngrams at least once."""

        features = {}
        counters = {}

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
                if counters[n][can] > self.n_gram_min_occurrences:
                    features[n].append(can)
                    begin = can[:-1]
                    end = can[1:]
                    if n > self.n_gram_min_length:
                        if counters[n - 1][begin] == counters[n][can] and begin in features[n - 1]:
                            features[n - 1].remove(begin)
                        if counters[n - 1][end] == counters[n][can] and end in features[n - 1]:
                            features[n - 1].remove(end)

        return [item for sublist in list(features.values()) for item in sublist]

    def _create_bow_vecs(self, intent_features, sentences, spacy_nlp, max_ngrams=None):
        """Given a set of sentences, returns a feature vector for each sentence.

        The first $k$ elements are from the `intent_features`,
        the rest are {1,0} elements denoting whether an ngram is in sentence."""

        import numpy as np

        ngrams_to_use = self._ngrams_to_use(max_ngrams)
        if ngrams_to_use is None:
            return intent_features
        extras = np.array(self._ngrams_in_sentences(sentences, spacy_nlp, ngrams=ngrams_to_use))
        total = np.hstack((intent_features, extras))
        return total

    def _cross_validation(self, sentences, labels, intent_features, spacy_nlp, max_ngrams):
        """choose the best number of ngrams to include in bow.

        Given an intent classification problem and a set of ordered ngrams (ordered in terms
        of importance by pick_applicable_ngrams) we choose the best number of ngrams to include
        in our bow vecs by cross validation."""

        from sklearn import preprocessing
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import cross_val_score
        import numpy as np

        clf2 = LogisticRegression()
        intent_encoder = preprocessing.LabelEncoder()
        intent_encoder.fit(labels)
        y = intent_encoder.transform(labels)
        cv_splits = min(10, np.min(np.bincount(y)))
        if cv_splits >= 3:
            logging.debug("Started ngram cross-validation to find best number of ngrams to use...")
            num_ngrams = np.unique(list(map(int, np.floor(np.linspace(1, max_ngrams, 8)))))
            no_ngrams_X = self._create_bow_vecs(intent_features, sentences, spacy_nlp, max_ngrams=0)
            no_ngrams_score = np.mean(cross_val_score(clf2, no_ngrams_X, y, cv=cv_splits))
            scores = []
            for n in num_ngrams:
                X = self._create_bow_vecs(intent_features, sentences, spacy_nlp, max_ngrams=n)
                score = np.mean(cross_val_score(clf2, X, y, cv=cv_splits))
                scores.append(score)
                logging.debug("Evaluating usage of {} ngrams. Score: {}".format(n, score))
            n_top = num_ngrams[np.argmax(scores)]
            logging.debug("Score without ngrams: {}".format(no_ngrams_score))
            logging.info("Best score with {} ngrams: {}".format(n_top, np.max(scores)))
            return n_top
        else:
            warnings.warn("Can't cross-validate ngram featurizer. There aren't enough examples per intent (at least 3)")
            return max_ngrams
