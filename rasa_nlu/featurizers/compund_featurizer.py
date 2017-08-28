from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import json
import logging
import warnings

import numpy as np
from builtins import str
import os
import io

from collections import defaultdict, Counter

from rasa_nlu.featurizers import Featurizer

logger = logging.getLogger(__name__)


class CompoundSplitter(object):
    """Split compound words into its parts. Based on the publication of

    'Splitting Compounds by Semantic Analogy'
    from Daiber, Joachim and Quiroz, Lautaro and Wechsler, Roger and Frank, Stella
    in Proceedings of the 1st Deep Machine Translation Workshop 2015

    This is a modified version of the implementation. the original can be found here:
    https://github.com/jodaiber/semantic_compound_splitting
    """

    def __init__(self, dict_filename):
        self.dict_filename = dict_filename
        self.splits = CompoundSplitter.load_dict(dict_filename)

    @staticmethod
    def load_dict(filename):
        """Loads a dictionary containing compound words and their split points."""

        splits = {}
        with open(filename) as f:
            for line in f:
                es = line.decode('utf8').rstrip('\n').split(" ")
                w = es[0].lower()
                indices = map(lambda i: i.split(','), es[1:])

                splits[w] = []
                for from_, to, fug in indices:
                    s, e = int(from_), int(to)
                    # Don't use single character splits - just add to prev split
                    if e - s == 1:
                        splits[w][-1][1] += 1
                    else:
                        splits[w].append([s, e, fug])
        return splits

    def split_token(self, token):
        """Splits a single token into its word parts (only if it is a known compound)."""

        if token.lower() in self.splits:
            splitted_parts = []
            for from_, to, fug in self.splits[token.lower()]:
                word_part = token[from_:to - len(fug)]
                splitted_parts.append(word_part)

            return splitted_parts
        else:
            return [token]

    def split(self, tokens):
        """Splits a collection of tokens by splitting each token on its own."""

        return [word for token in tokens for word in self.split_token(token)]


class SynonymTokenMapper(object):
    """Replaces unknown words in a document with suitable and known synonyms."""

    def __init__(self, dict_filename):
        self.dict_filename = dict_filename
        self.mappings = SynonymTokenMapper.load_dict(dict_filename)

    @staticmethod
    def load_dict(filename):
        """Loads a synonym dictionary from file. The file should contain lines with two words per line.

        The first word will be replaced by the second one."""

        mapping = {}
        with open(filename) as f:
            for line in f:
                es = line.decode('utf8').rstrip('\n').split(" ")
                word = es[0].lower()
                mapped_to = es[1].lower()
                mapping[word] = mapped_to
        return mapping

    def synonym_for(self, token):
        """Try to find a synonym in the dictionary for the token."""

        return self.mappings.get(token.lower(), token)


class SpacyCompoundFeaturizer(Featurizer):
    """Provides a featurizer that includes compound splitting and synonym replacement capabilities."""

    name = "intent_featurizer_compounds"

    provides = ["text_features"]

    requires = ["spacy_doc"]

    def __init__(self, splitter=None, synonym_mapper=None):
        self.splitter = splitter
        self.synonym_mapper = synonym_mapper

    def load_from_dicts(self, compound_dict_filename=None, synonym_dict_filename=None):
        self.splitter = CompoundSplitter(compound_dict_filename) if compound_dict_filename else None
        self.synonym_mapper = SynonymTokenMapper(synonym_dict_filename) if synonym_dict_filename else None

    def train(self, training_data, config, **kwargs):
        # type: (TrainingData) -> None

        self.load_from_dicts(config["compound_dict"], config["synonym_dict"])
        spacy_nlp = kwargs.get("spacy_nlp", None)
        unknowns = Counter()

        for example in training_data.intent_examples:
            features, unkns = self.create_bow_vecs(example, spacy_nlp)
            for u in unkns:
                unknowns[u] += 1
            example.set("text_features", self._combine_with_existing_text_features(example, features))

        for u, c in unknowns.most_common():
            logger.debug("Unknown token: '{}' Occurences: {}".format(u, c))

    def process(self, message, **kwargs):
        # type: (Message, **Any) -> None

        spacy_nlp = kwargs.get("spacy_nlp", None)
        features, _ = self.create_bow_vecs(message, spacy_nlp)
        message.set("text_features", self._combine_with_existing_text_features(message, features))

    def split_invalid_tokens(self, token):
        """In case we do not know a token (i.e. word vector is all 0's) it is most likely a compound we need to split.

        If a word has a nonzero word vector, it will not get split."""

        if not token.has_vector:
            return self.splitter.split_token(token.text)
        else:
            return [token.text]

    def synonym_for(self, token_text):
        if self.synonym_mapper:
            return self.synonym_mapper.synonym_for(token_text.lower())
        else:
            return token_text.lower()

    def create_bow_vecs(self, message, spacy_nlp):
        """Transform the sentences into matrix representation."""

        # We need to do two passes over the document:
        #    First pass: to collect all tokens of the sentence
        #    Second pass: if there are tokens that are likely compounds, they will be split and the sentence with
        #                 the split compounds is passed through the NLP pipeline again
        doc = message.get("spacy_doc")
        sentence_with_split_compounds = \
            u" ".join([self.synonym_for(part) for token in doc for part in self.split_invalid_tokens(token)])
        doc = spacy_nlp(sentence_with_split_compounds)
        vec = []
        unknowns = []
        for token in doc:
            if not token.has_vector:
                if not (token.is_punct or token.is_digit or token.is_space):
                    unknowns.append(token.text)
            else:
                vec.append(token.vector)

        if vec:
            return np.sum(vec, axis=0) / len(vec), unknowns
        else:
            return doc.vector, unknowns

    @classmethod
    def load(cls, model_dir=None, model_metadata=None, cached_component=None, **kwargs):
        # type: (Text, Metadata, Optional[Component], **Any) -> SpacyCompoundFeaturizer

        if model_dir and model_metadata.get("compound_dict"):
            compound_file = os.path.join(model_dir, model_metadata.get("compound_dict"))
            synonyms_file = os.path.join(model_dir, model_metadata.get("compound_synonyms"))
            featurizer = SpacyCompoundFeaturizer()
            featurizer.load_from_dicts(compound_file, synonyms_file)
            return featurizer
        else:
            return SpacyCompoundFeaturizer()

    def persist(self, model_dir):
        # type: (Text) -> Dict[Text, Any]
        """Persist this model into the passed directory. Returns the metadata necessary to load the model again."""
        from shutil import copyfile

        classifier_file = os.path.join(model_dir, "compound_featurizer.json")
        if self.splitter:
            splitter_file_name = "splitter.dict"
            copyfile(self.splitter.dict_filename, os.path.join(model_dir, splitter_file_name))
        else:
            splitter_file_name = None

        if self.synonym_mapper:
            syonyms_file_name = "synonyms.dict"
            copyfile(self.synonym_mapper.dict_filename, os.path.join(model_dir, syonyms_file_name))
        else:
            syonyms_file_name = None
        return {
                "compound_dict": splitter_file_name,
                "compound_synonyms": syonyms_file_name
            }
