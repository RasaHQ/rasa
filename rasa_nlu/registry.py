"""This is a somewhat delicate package. It contains all registered components and preconfigured templates.

Hence, it imports all of the components. To avoid cycles, no component should import this in module scope."""
from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from typing import Optional
from typing import Type

from rasa_nlu.classifiers.keyword_intent_classifier import KeywordIntentClassifier
from rasa_nlu.classifiers.mitie_intent_classifier import MitieIntentClassifier
from rasa_nlu.classifiers.sklearn_intent_classifier import SklearnIntentClassifier
from rasa_nlu.extractors.entity_synonyms import EntitySynonymMapper
from rasa_nlu.extractors.mitie_entity_extractor import MitieEntityExtractor
from rasa_nlu.extractors.spacy_entity_extractor import SpacyEntityExtractor
from rasa_nlu.featurizers.mitie_featurizer import MitieFeaturizer
from rasa_nlu.featurizers.ngram_featurizer import NGramFeaturizer
from rasa_nlu.featurizers.spacy_featurizer import SpacyFeaturizer
from rasa_nlu.tokenizers.mitie_tokenizer import MitieTokenizer
from rasa_nlu.utils.mitie_utils import MitieNLP
from rasa_nlu.utils.spacy_utils import SpacyNLP

# Classes of all known components. If a new component should be added, its class needs to be listed here.
component_classes = [
    SpacyNLP, SpacyEntityExtractor, SklearnIntentClassifier, SpacyFeaturizer,
    MitieNLP, MitieEntityExtractor, MitieIntentClassifier, MitieFeaturizer, MitieTokenizer,
    KeywordIntentClassifier, EntitySynonymMapper, NGramFeaturizer]

# Mapping from a components name to its class to allow name based lookup.
registered_components = {component.name: component for component in component_classes}

# To simplify usage, there are a couple of model templates, that already add necessary components in the right order.
# They also implement the preexisting `backends`.
registered_pipeline_templates = {
    "spacy_sklearn": [
        "init_spacy",
        "ner_spacy",
        "ner_synonyms",
        "intent_featurizer_spacy",
        "intent_classifier_sklearn",
    ],
    "mitie": [
        "init_mitie",
        "tokenizer_mitie",
        "ner_mitie",
        "ner_synonyms",
        "intent_classifier_mitie",
    ],
    "mitie_sklearn": [
        "init_mitie",
        "tokenizer_mitie",
        "ner_mitie",
        "ner_synonyms",
        "intent_featurizer_mitie",
        "intent_classifier_sklearn",
    ],
    "keyword": [
        "intent_classifier_keyword",
    ]
}


def get_component_class(component_name):
    # type: (str) -> Optional[Type[Component]]
    """Resolve component name to a registered components class."""
    from rasa_nlu.components import Component

    return registered_components.get(component_name)


def load_component_by_name(component_name, context, config):
    # type: (str, dict, dict) -> Optional[Component]
    """Resolves a components name and calls it's load method to init it based on a previously persisted model."""
    from rasa_nlu.components import load_component
    from rasa_nlu.components import Component

    component_clz = get_component_class(component_name)
    return load_component(component_clz, context, config)
