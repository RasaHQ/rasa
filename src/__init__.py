import codecs
import json
import os
import warnings

import version
from rasa_nlu.classifiers.mitie_intent_classifier import MitieIntentClassifier
from rasa_nlu.classifiers.sklearn_intent_classifier import SklearnIntentClassifier
from rasa_nlu.extractors.mitie_entity_extractor import MitieEntityExtractor
from rasa_nlu.extractors.spacy_entity_extractor import SpacyEntityExtractor
from rasa_nlu.featurizers.mitie_featurizer import MitieFeaturizer
from rasa_nlu.featurizers.spacy_featurizer import SpacyFeaturizer
from rasa_nlu.tokenizers.mitie_tokenizer import MitieTokenizer
from rasa_nlu.utils.mitie_utils import MitieNLP
from rasa_nlu.utils.spacy_utils import SpacyNLP

__version__ = version.__version__


