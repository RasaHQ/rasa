"""This is a somewhat delicate package. It contains all registered components
and preconfigured templates.

Hence, it imports all of the components. To avoid cycles, no component should
import this in module scope."""

import logging
import typing
from typing import Any, Dict, List, Optional, Text, Type

from rasa_nlu import utils
from rasa_nlu.classifiers.embedding_intent_classifier import \
    EmbeddingIntentClassifier
from rasa_nlu.classifiers.keyword_intent_classifier import \
    KeywordIntentClassifier
from rasa_nlu.classifiers.mitie_intent_classifier import MitieIntentClassifier
from rasa_nlu.classifiers.sklearn_intent_classifier import \
    SklearnIntentClassifier
from rasa_nlu.extractors.crf_entity_extractor import CRFEntityExtractor
from rasa_nlu.extractors.duckling_http_extractor import DucklingHTTPExtractor
from rasa_nlu.extractors.entity_synonyms import EntitySynonymMapper
from rasa_nlu.extractors.mitie_entity_extractor import MitieEntityExtractor
from rasa_nlu.extractors.spacy_entity_extractor import SpacyEntityExtractor
from rasa_nlu.featurizers.count_vectors_featurizer import \
    CountVectorsFeaturizer
from rasa_nlu.featurizers.mitie_featurizer import MitieFeaturizer
from rasa_nlu.featurizers.ngram_featurizer import NGramFeaturizer
from rasa_nlu.featurizers.regex_featurizer import RegexFeaturizer
from rasa_nlu.featurizers.spacy_featurizer import SpacyFeaturizer
from rasa_nlu.model import Metadata
from rasa_nlu.tokenizers.jieba_tokenizer import JiebaTokenizer
from rasa_nlu.tokenizers.mitie_tokenizer import MitieTokenizer
from rasa_nlu.tokenizers.spacy_tokenizer import SpacyTokenizer
from rasa_nlu.tokenizers.whitespace_tokenizer import WhitespaceTokenizer
from rasa_nlu.utils.mitie_utils import MitieNLP
from rasa_nlu.utils.spacy_utils import SpacyNLP

if typing.TYPE_CHECKING:
    from rasa_nlu.components import Component
    from rasa_nlu.config import RasaNLUModelConfig, RasaNLUModelConfig

logger = logging.getLogger(__name__)


# Classes of all known components. If a new component should be added,
# its class name should be listed here.
component_classes = [
    # utils
    SpacyNLP, MitieNLP,
    # tokenizers
    MitieTokenizer, SpacyTokenizer, WhitespaceTokenizer, JiebaTokenizer,
    # extractors
    SpacyEntityExtractor, MitieEntityExtractor, CRFEntityExtractor,
    DucklingHTTPExtractor, EntitySynonymMapper,
    # featurizers
    SpacyFeaturizer, MitieFeaturizer, NGramFeaturizer, RegexFeaturizer,
    CountVectorsFeaturizer,
    # classifiers
    SklearnIntentClassifier, MitieIntentClassifier, KeywordIntentClassifier,
    EmbeddingIntentClassifier
]

# Mapping from a components name to its class to allow name based lookup.
registered_components = {c.name: c for c in component_classes}

# DEPRECATED ensures compatibility, will be remove in future versions
old_style_names = {
    "nlp_spacy": "SpacyNLP",
    "nlp_mitie": "MitieNLP",
    "ner_spacy": "SpacyEntityExtractor",
    "ner_mitie": "MitieEntityExtractor",
    "ner_crf": "CRFEntityExtractor",
    "ner_duckling_http": "DucklingHTTPExtractor",
    "ner_synonyms": "EntitySynonymMapper",
    "intent_featurizer_spacy": "SpacyFeaturizer",
    "intent_featurizer_mitie": "MitieFeaturizer",
    "intent_featurizer_ngrams": "NGramFeaturizer",
    "intent_entity_featurizer_regex": "RegexFeaturizer",
    "intent_featurizer_count_vectors": "CountVectorsFeaturizer",
    "tokenizer_mitie": "MitieTokenizer",
    "tokenizer_spacy": "SpacyTokenizer",
    "tokenizer_whitespace": "WhitespaceTokenizer",
    "tokenizer_jieba": "JiebaTokenizer",
    "intent_classifier_sklearn": "SklearnIntentClassifier",
    "intent_classifier_mitie": "MitieIntentClassifier",
    "intent_classifier_keyword": "KeywordIntentClassifier",
    "intent_classifier_tensorflow_embedding": "EmbeddingIntentClassifier"
}

# To simplify usage, there are a couple of model templates, that already add
# necessary components in the right order. They also implement
# the preexisting `backends`.
registered_pipeline_templates = {
    "pretrained_embeddings_spacy": [
        "SpacyNLP",
        "SpacyTokenizer",
        "SpacyFeaturizer",
        "RegexFeaturizer",
        "CRFEntityExtractor",
        "EntitySynonymMapper",
        "SklearnIntentClassifier",
    ],
    "keyword": [
        "KeywordIntentClassifier",
    ],
    "supervised_embeddings": [
        "WhitespaceTokenizer",
        "RegexFeaturizer",
        "CRFEntityExtractor",
        "EntitySynonymMapper",
        "CountVectorsFeaturizer",
        "EmbeddingIntentClassifier"
    ]
}


def pipeline_template(s: Text) -> Optional[List[Dict[Text, Text]]]:
    components = registered_pipeline_templates.get(s)

    if components:
        # converts the list of components in the configuration
        # format expected (one json object per component)
        return [{"name": c} for c in components]

    else:
        return None


def get_component_class(component_name: Text) -> Type['Component']:
    """Resolve component name to a registered components class."""

    if component_name not in registered_components:
        if component_name not in old_style_names:
            try:
                return utils.class_from_module_path(component_name)
            except Exception:
                raise Exception(
                    "Failed to find component class for '{}'. Unknown "
                    "component name. Check your configured pipeline and make "
                    "sure the mentioned component is not misspelled. If you "
                    "are creating your own component, make sure it is either "
                    "listed as part of the `component_classes` in "
                    "`rasa_nlu.registry.py` or is a proper name of a class "
                    "in a module.".format(component_name))
        else:
            # DEPRECATED ensures compatibility, remove in future versions
            logger.warning("DEPRECATION warning: your nlu config file "
                           "contains old style component name `{}`, "
                           "you should change it to its class name: `{}`."
                           "".format(component_name,
                                     old_style_names[component_name]))
            component_name = old_style_names[component_name]

    return registered_components[component_name]


def load_component_by_meta(component_meta: Dict[Text, Any],
                           model_dir: Text,
                           metadata: Metadata,
                           cached_component: Optional['Component'],
                           **kwargs: Any
                           ) -> Optional['Component']:
    """Resolves a component and calls its load method.

    Inits it based on a previously persisted model.
    """

    # try to get class name first, else create by name
    component_name = component_meta.get('class', component_meta['name'])
    component_class = get_component_class(component_name)
    return component_class.load(component_meta, model_dir, metadata,
                                cached_component, **kwargs)


def create_component_by_config(component_config: Dict[Text, Any],
                               config: 'RasaNLUModelConfig'
                               ) -> Optional['Component']:
    """Resolves a component and calls it's create method.

    Inits it based on a previously persisted model.
    """

    # try to get class name first, else create by name
    component_name = component_config.get('class', component_config['name'])
    component_class = get_component_class(component_name)
    return component_class.create(component_config, config)
