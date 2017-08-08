from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import logging
import os

import typing
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Text
from typing import Tuple

from rasa_nlu.config import RasaNLUConfig
from rasa_nlu.extractors import EntityExtractor
from rasa_nlu.model import Metadata
from rasa_nlu.tokenizers import Token
from rasa_nlu.training_data import Message
from rasa_nlu.training_data import TrainingData
from builtins import str

logger = logging.getLogger(__name__)

if typing.TYPE_CHECKING:
    from spacy.language import Language
    import sklearn_crfsuite
    from spacy.tokens import Doc


class CRFEntityExtractor(EntityExtractor):
    name = "ner_crf"

    provides = ["entities"]

    requires = ["spacy_doc", "tokens"]

    function_dict = {
        'low': lambda doc: doc[0].lower(),
        'title': lambda doc: doc[0].istitle(),
        'word3': lambda doc: doc[0][-3:],
        'word2': lambda doc: doc[0][-2:],
        'pos': lambda doc: doc[1],
        'pos2': lambda doc: doc[1][:2],
        'bias': lambda doc: 'bias',
        'upper': lambda doc: doc[0].isupper(),
        'digit': lambda doc: doc[0].isdigit(),
        'pattern': lambda doc: str(doc[3]) if doc[3] is not None else 'N/A',
    }

    def __init__(self, ent_tagger=None, entity_crf_features=None, entity_crf_BILOU_flag=True):
        # type: (sklearn_crfsuite.CRF, List[List[Text]], bool) -> None

        self.ent_tagger = ent_tagger

        # BILOU_flag determines whether to use BILOU tagging or not.
        # More rigorous however requires more examples per entity
        # rule of thumb: use only if more than 100 egs. per entity
        self.BILOU_flag = entity_crf_BILOU_flag

        if not entity_crf_features:
            # crf_features is [before, word, after] array with before, word, after holding keys about which
            # features to use for each word, for example, 'title' in array before will have the feature
            # "is the preceding word in title case?"
            self.crf_features = [
                ['low', 'title', 'upper', 'pos', 'pos2'],
                ['bias', 'low', 'word3', 'word2', 'upper', 'title', 'digit', 'pos', 'pos2', 'pattern'],
                ['low', 'title', 'upper', 'pos', 'pos2']
            ]
        else:
            self.crf_features = entity_crf_features

    @classmethod
    def required_packages(cls):
        return ["sklearn_crfsuite", "sklearn", "spacy"]

    def train(self, training_data, config, **kwargs):
        # type: (TrainingData, RasaNLUConfig) -> None

        train_config = config.get("ner_crf", {})

        # These two are expected to be in the config so not using .get
        self.BILOU_flag = train_config["BILOU_flag"]
        self.crf_features = train_config["features"]

        self.max_iterations = train_config.get("max_iterations", 50)
        self.L1_C = config.get("L1_c", 1)
        self.L2_C = config.get("L2_c", 1e-3)

        if training_data.entity_examples:
            # convert the dataset into features
            dataset = self._create_dataset(training_data.entity_examples)
            # train the model
            self._train_model(dataset)

    def _create_dataset(self, examples):
        # type: (List[Message]) -> List[List[Tuple[Text, Text, Text, Text]]]
        dataset = []
        for example in examples:
            entity_offsets = self._convert_example(example)
            dataset.append(self._from_json_to_crf(example, entity_offsets))
        return dataset

    def test(self, testing_data):
        # type: (TrainingData, Language) -> None

        if testing_data.num_entity_examples > 0:
            dataset = self._create_dataset(testing_data.entity_examples)
            self._test_model(dataset)

    def process(self, message, **kwargs):
        # type: (Message, **Any) -> None

        extracted = self.add_extractor_name(self.extract_entities(message))
        message.set("entities", message.get("entities", []) + extracted, add_to_output=True)

    def _convert_example(self, example):
        # type: (Message) -> List[Tuple[int, int, Text]]

        def convert_entity(ent):
            return ent["start"], ent["end"], ent["entity"]

        return [convert_entity(ent) for ent in example.get("entities", [])]

    def extract_entities(self, message):
        # type: (Message) -> List[Dict[Text, Any]]
        """Take a sentence and return entities in json format"""

        if self.ent_tagger is not None:
            text_data = self._from_text_to_crf(message)
            features = self._sentence_to_features(text_data)
            ents = self.ent_tagger.predict_single(features)
            return self._from_crf_to_json(message, ents)
        else:
            return []

    def _from_crf_to_json(self, message, entities):
        # type: (Message, List[Any]) -> List[Dict[Text, Any]]

        sentence_doc = message.get("spacy_doc")
        json_ents = []
        if len(sentence_doc) != len(entities):
            raise Exception('Inconsistency in amount of tokens between crfsuite and spacy')
        if self.BILOU_flag:
            # using the BILOU tagging scheme
            for word_idx in range(len(sentence_doc)):
                entity = entities[word_idx]
                word = sentence_doc[word_idx]
                if entity.startswith('U-'):
                    ent = {'start': word.idx, 'end': word.idx + len(word),
                           'value': word.text, 'entity': entity[2:]}
                    json_ents.append(ent)
                elif entity.startswith('B-'):
                    # start of a multi-word entity, need to represent whole extent
                    ent_word_idx = word_idx + 1
                    finished = False
                    while not finished:
                        if len(entities) > ent_word_idx and entities[ent_word_idx][2:] != entity[2:]:
                            # words are not tagged the same entity class
                            logger.debug(
                                    "Inconsistent BILOU tagging found, B- tag, L- tag pair encloses multiple " +
                                    "entity classes.i.e. ['B-a','I-b','L-a'] instead of ['B-a','I-a','L-a'].\n" +
                                    "Assuming B- class is correct.")
                        if len(entities) > ent_word_idx and entities[ent_word_idx].startswith('L-'):
                            # end of the entity
                            finished = True
                        elif len(entities) > ent_word_idx and entities[ent_word_idx].startswith('I-'):
                            # middle part of the entity
                            ent_word_idx += 1
                        else:
                            # entity not closed by an L- tag
                            finished = True
                            ent_word_idx -= 1
                            logger.debug(
                                    "Inconsistent BILOU tagging found, B- tag not closed by L- tag, " +
                                    "i.e ['B-a','I-a','O'] instead of ['B-a','L-a','O'].\nAssuming last tag is L-")
                    ent = {'start': word.idx,
                           'end': sentence_doc[word_idx:ent_word_idx + 1].end_char,
                           'value': sentence_doc[word_idx:ent_word_idx + 1].text,
                           'entity': entity[2:]}
                    json_ents.append(ent)
        elif not self.BILOU_flag:
            # not using BILOU tagging scheme, multi-word entities are split.
            for word_idx in range(len(sentence_doc)):
                entity = entities[word_idx]
                word = sentence_doc[word_idx]
                if entity != 'O':
                    ent = {'start': word.idx,
                           'end': word.idx + len(word),
                           'value': word.text,
                           'entity': entity}
                    json_ents.append(ent)
        return json_ents

    @classmethod
    def load(cls, model_dir, model_metadata, cached_component, **kwargs):
        # type: (Text, Metadata, Optional[CRFEntityExtractor], **Any) -> CRFEntityExtractor
        from sklearn.externals import joblib

        if model_dir and model_metadata.get("entity_extractor_crf"):
            meta = model_metadata.get("entity_extractor_crf")
            ent_tagger = joblib.load(os.path.join(model_dir, meta["model_file"]))
            return CRFEntityExtractor(ent_tagger=ent_tagger,
                                      entity_crf_features=meta['crf_features'],
                                      entity_crf_BILOU_flag=meta['BILOU_flag'])
        else:
            return CRFEntityExtractor()

    def persist(self, model_dir):
        # type: (Text) -> Dict[Text, Any]
        """Persist this model into the passed directory. Returns the metadata necessary to load the model again."""
        from sklearn.externals import joblib

        if self.ent_tagger:
            model_file_name = os.path.join(model_dir, "crf_model.pkl")

            joblib.dump(self.ent_tagger, model_file_name)
            return {"entity_extractor_crf": {"model_file": "crf_model.pkl",
                                             "crf_features": self.crf_features,
                                             "BILOU_flag": self.BILOU_flag,
                                             "version": 1}}
        else:
            return {"entity_extractor_crf": None}

    def _sentence_to_features(self, sentence):
        # type: (List[Tuple[Text, Text, Text, Text]]) -> List[Dict[Text, Any]]
        """Convert a word into discrete features in self.crf_features, including word before and word after."""

        sentence_features = []
        for word_idx in range(len(sentence)):
            # word before(-1), current word(0), next word(+1)
            prefixes = ['-1', '0', '+1']
            word_features = {}
            for i in range(3):
                if word_idx == len(sentence) - 1 and i == 2:
                    word_features['EOS'] = True
                    # End Of Sentence
                elif word_idx == 0 and i == 0:
                    word_features['BOS'] = True
                    # Beginning Of Sentence
                else:
                    word = sentence[word_idx - 1 + i]
                    prefix = prefixes[i]
                    features = self.crf_features[i]
                    for feature in features:
                        # append each feature to a feature vector
                        # word_features.append(prefix + feature + ':' + self.function_dict[feature](word))
                        word_features[prefix + ":" + feature] = self.function_dict[feature](word)
            sentence_features.append(word_features)
        return sentence_features

    def _sentence_to_labels(self, sentence):
        # type: (List[Tuple[Text, Text, Text, Text]]) -> List[Text]

        return [label for _, _, label, _ in sentence]

    def _from_json_to_crf(self, message, entity_offsets):
        # type: (Message, List[Tuple[int, int, Text]]) -> List[Tuple[Text, Text, Text, Text]]
        """Takes the json examples and switches them to a format which crfsuite likes."""
        from spacy.gold import GoldParse

        doc = message.get("spacy_doc")
        gold = GoldParse(doc, entities=entity_offsets)
        ents = [l[5] for l in gold.orig_annot]
        if '-' in ents:
            logger.warn("Misaligned entity annotation in sentence '{}'. ".format(doc.text) +
                        "Make sure the start and end values of the annotated training " +
                        "examples end at token boundaries (e.g. don't include trailing whitespaces).")
        if not self.BILOU_flag:
            for i, entity in enumerate(ents):
                if entity.startswith('B-') or \
                        entity.startswith('I-') or \
                        entity.startswith('U-') or \
                        entity.startswith('L-'):
                    ents[i] = entity[2:]        # removes the BILOU tags

        return self._from_text_to_crf(message, ents)

    def __pattern_of_token(self, message, i):
        if message.get("tokens"):
            return message.get("tokens")[i].get("pattern")
        else:
            return None

    def _from_text_to_crf(self, message, entities=None):
        # type: (Message, List[Text]) -> List[Tuple[Text, Text, Text, Text]]
        """Takes a sentence and switches it to crfsuite format."""

        crf_format = []
        for i, token in enumerate(message.get("spacy_doc")):
            pattern = self.__pattern_of_token(message, i)
            entity = entities[i] if entities else "N/A"
            crf_format.append((token.text, token.tag_, entity, pattern))
        return crf_format

    def _train_model(self, df_train):
        # type: (List[List[Tuple[Text, Text, Text, Text]]]) -> None
        """Train the crf tagger based on the training data."""
        import sklearn_crfsuite

        X_train = [self._sentence_to_features(sent) for sent in df_train]
        y_train = [self._sentence_to_labels(sent) for sent in df_train]
        self.ent_tagger = sklearn_crfsuite.CRF(
                algorithm='lbfgs',
                c1=self.L1_C,  # coefficient for L1 penalty
                c2=self.L2_C,  # coefficient for L2 penalty
                max_iterations=self.max_iterations,  # stop earlier
                all_possible_transitions=True  # include transitions that are possible, but not observed
        )
        self.ent_tagger.fit(X_train, y_train)

    def _test_model(self, df_test):
        # type: (List[List[Tuple[Text, Text, Text, Text]]]) -> None

        X_test = [self._sentence_to_features(sent) for sent in df_test]
        y_test = [self._sentence_to_labels(sent) for sent in df_test]
        y_pred = [self.ent_tagger.predict_single(xseq) for xseq in X_test]
        print(bio_classification_report(y_test, y_pred))


def bio_classification_report(y_true, y_pred):
    """Evaluates entity extraction accuracy.

    Classification report for a list of BIO-encoded sequences.
    It computes token-level metrics and discards "O" labels.
    Note that it requires scikit-learn 0.15+ (or a version from github master)
    to calculate averages properly!
    Taken from https://github.com/scrapinghub/python-crfsuite/blob/master/examples/CoNLL%202002.ipynb
    """
    from sklearn.preprocessing import LabelBinarizer
    from itertools import chain
    from sklearn.metrics import classification_report

    lb = LabelBinarizer()
    y_true_combined = lb.fit_transform(list(chain.from_iterable(y_true)))
    y_pred_combined = lb.transform(list(chain.from_iterable(y_pred)))

    tagset = set(lb.classes_) - {'O'}
    tagset = sorted(tagset, key=lambda tag: tag.split('-', 1)[::-1])
    class_indices = {cls: idx for idx, cls in enumerate(lb.classes_)}

    return classification_report(
            y_true_combined,
            y_pred_combined,
            labels=[class_indices[cls] for cls in tagset],
            target_names=tagset,
    )
