"""This is a somewhat delicate package. It contains all registered components and preconfigured templates.

Hence, it imports all of the components. To avoid cycles, no component should import this in module scope."""
from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import typing
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Text
from typing import Type

from rasa_nlu.classifiers.keyword_intent_classifier import KeywordIntentClassifier
from rasa_nlu.classifiers.mitie_intent_classifier import MitieIntentClassifier
from rasa_nlu.classifiers.sklearn_intent_classifier import SklearnIntentClassifier
from rasa_nlu.extractors.duckling_extractor import DucklingExtractor
from rasa_nlu.extractors.entity_synonyms import EntitySynonymMapper
from rasa_nlu.extractors.mitie_entity_extractor import MitieEntityExtractor
from rasa_nlu.extractors.spacy_entity_extractor import SpacyEntityExtractor
from rasa_nlu.extractors.crf_entity_extractor import CRFEntityExtractor
from rasa_nlu.featurizers.mitie_featurizer import MitieFeaturizer
from rasa_nlu.featurizers.ngram_featurizer import NGramFeaturizer
from rasa_nlu.featurizers.regex_featurizer import RegexFeaturizer
from rasa_nlu.featurizers.spacy_featurizer import SpacyFeaturizer
from rasa_nlu.model import Metadata
from rasa_nlu.tokenizers.mitie_tokenizer import MitieTokenizer
from rasa_nlu.tokenizers.spacy_tokenizer import SpacyTokenizer
from rasa_nlu.tokenizers.whitespace_tokenizer import WhitespaceTokenizer
from rasa_nlu.utils.mitie_utils import MitieNLP
from rasa_nlu.utils.spacy_utils import SpacyNLP

if typing.TYPE_CHECKING:
    from rasa_nlu.components import Component
    from rasa_nlu.config import RasaNLUConfig

# Classes of all known components. If a new component should be added, its class needs to be listed here.
component_classes = [
    SpacyNLP, MitieNLP,
    SpacyEntityExtractor, MitieEntityExtractor, DucklingExtractor, CRFEntityExtractor,
    EntitySynonymMapper,
    SpacyFeaturizer, MitieFeaturizer, NGramFeaturizer, RegexFeaturizer,
    MitieTokenizer, SpacyTokenizer, WhitespaceTokenizer,
    SklearnIntentClassifier, MitieIntentClassifier, KeywordIntentClassifier,
]

# Mapping from a components name to its class to allow name based lookup.
registered_components = {
    component.name: component for component in component_classes}  # type: Dict[Text, Type[Component]]

# To simplify usage, there are a couple of model templates, that already add necessary components in the right order.
# They also implement the preexisting `backends`.
registered_pipeline_templates = {
    "spacy_sklearn": [
        "nlp_spacy",
        "tokenizer_spacy",
        "intent_featurizer_spacy",
        "intent_entity_featurizer_regex",
        "ner_crf",
        "ner_synonyms",
        "intent_classifier_sklearn",
    ],
    "mitie": [
        "nlp_mitie",
        "tokenizer_mitie",
        "ner_mitie",
        "ner_synonyms",
        "intent_entity_featurizer_regex",
        "intent_classifier_mitie",
    ],
    "mitie_sklearn": [
        "nlp_mitie",
        "tokenizer_mitie",
        "ner_mitie",
        "ner_synonyms",
        "intent_entity_featurizer_regex",
        "intent_featurizer_mitie",
        "intent_classifier_sklearn",
    ],
    "keyword": [
        "intent_classifier_keyword",
    ],
    # this template really is just for testing
    # every component should be in here so train-persist-load-use cycle can be tested
    # they still need to be in a useful order - hence we can not simply generate this automatically
    "all_components": [
        "nlp_spacy",
        "nlp_mitie",
        "tokenizer_whitespace",
        "tokenizer_mitie",
        "tokenizer_spacy",
        "intent_featurizer_mitie",
        "intent_featurizer_spacy",
        "intent_featurizer_ngrams",
        "intent_entity_featurizer_regex",
        "ner_mitie",
        "ner_crf",
        "ner_spacy",
        "ner_duckling",
        "ner_synonyms",
        "intent_classifier_keyword",
        "intent_classifier_sklearn",
        "intent_classifier_mitie",
    ]
}


def get_component_class(component_name):
    # type: (Text) -> Optional[Type[Component]]
    """Resolve component name to a registered components class."""

    if component_name not in registered_components:
        raise Exception("Failed to find component class for '{}'. Unknown component name. ".format(component_name) +
                        "Check your configured pipeline and make sure the mentioned component is not misspelled. " +
                        "If you are creating your own component, make sure it is listed as part of the " +
                        "`component_classes` in `rasa_nlu.registry.py`.")
    return registered_components[component_name]


def load_component_by_name(component_name, model_dir, metadata, cached_component, **kwargs):
    # type: (Text, Text, Metadata, Optional[Component], **Any) -> Optional[Component]
    """Resolves a components name and calls it's load method to init it based on a previously persisted model."""

    component_clz = get_component_class(component_name)
    return component_clz.load(model_dir, metadata, cached_component, **kwargs)


def create_component_by_name(component_name, config):
    # type: (Text, RasaNLUConfig) -> Optional[Component]
    """Resolves a components name and calls it's create method to init it based on a previously persisted model."""

    component_clz = get_component_class(component_name)
    return component_clz.create(config)
