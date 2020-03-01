"""This is a somewhat delicate package. It contains all registered components
and preconfigured templates.

Hence, it imports all of the components. To avoid cycles, no component should
import this in module scope."""

import logging
import typing
from typing import Any, Dict, List, Optional, Text, Type

from rasa.constants import DOCS_URL_COMPONENTS

from rasa.nlu.classifiers.diet_classifier import DIETClassifier
from rasa.nlu.classifiers.keyword_intent_classifier import KeywordIntentClassifier
from rasa.nlu.classifiers.mitie_intent_classifier import MitieIntentClassifier
from rasa.nlu.classifiers.sklearn_intent_classifier import SklearnIntentClassifier
from rasa.nlu.classifiers.embedding_intent_classifier import EmbeddingIntentClassifier
from rasa.nlu.extractors.crf_entity_extractor import CRFEntityExtractor
from rasa.nlu.extractors.duckling_http_extractor import DucklingHTTPExtractor
from rasa.nlu.extractors.entity_synonyms import EntitySynonymMapper
from rasa.nlu.extractors.mitie_entity_extractor import MitieEntityExtractor
from rasa.nlu.extractors.spacy_entity_extractor import SpacyEntityExtractor
from rasa.nlu.featurizers.sparse_featurizer.lexical_syntactic_featurizer import (
    LexicalSyntacticFeaturizer,
)
from rasa.nlu.featurizers.dense_featurizer.convert_featurizer import ConveRTFeaturizer
from rasa.nlu.featurizers.dense_featurizer.mitie_featurizer import MitieFeaturizer
from rasa.nlu.featurizers.dense_featurizer.spacy_featurizer import SpacyFeaturizer
from rasa.nlu.featurizers.sparse_featurizer.count_vectors_featurizer import (
    CountVectorsFeaturizer,
)
from rasa.nlu.featurizers.dense_featurizer.lm_featurizer import LanguageModelFeaturizer
from rasa.nlu.featurizers.sparse_featurizer.regex_featurizer import RegexFeaturizer
from rasa.nlu.model import Metadata
from rasa.nlu.selectors.response_selector import ResponseSelector
from rasa.nlu.tokenizers.convert_tokenizer import ConveRTTokenizer
from rasa.nlu.tokenizers.jieba_tokenizer import JiebaTokenizer
from rasa.nlu.tokenizers.mitie_tokenizer import MitieTokenizer
from rasa.nlu.tokenizers.spacy_tokenizer import SpacyTokenizer
from rasa.nlu.tokenizers.whitespace_tokenizer import WhitespaceTokenizer
from rasa.nlu.tokenizers.lm_tokenizer import LanguageModelTokenizer
from rasa.nlu.utils.mitie_utils import MitieNLP
from rasa.nlu.utils.spacy_utils import SpacyNLP
from rasa.nlu.utils.hugging_face.hf_transformers import HFTransformersNLP
from rasa.utils.common import class_from_module_path, raise_warning
from rasa.utils.tensorflow.constants import (
    INTENT_CLASSIFICATION,
    ENTITY_RECOGNITION,
    NUM_TRANSFORMER_LAYERS,
)

if typing.TYPE_CHECKING:
    from rasa.nlu.components import Component
    from rasa.nlu.config import RasaNLUModelConfig, RasaNLUModelConfig

logger = logging.getLogger(__name__)


# Classes of all known components. If a new component should be added,
# its class name should be listed here.
component_classes = [
    # utils
    SpacyNLP,
    MitieNLP,
    HFTransformersNLP,
    # tokenizers
    MitieTokenizer,
    SpacyTokenizer,
    WhitespaceTokenizer,
    ConveRTTokenizer,
    JiebaTokenizer,
    LanguageModelTokenizer,
    # extractors
    SpacyEntityExtractor,
    MitieEntityExtractor,
    CRFEntityExtractor,
    DucklingHTTPExtractor,
    EntitySynonymMapper,
    # featurizers
    SpacyFeaturizer,
    MitieFeaturizer,
    RegexFeaturizer,
    LexicalSyntacticFeaturizer,
    CountVectorsFeaturizer,
    ConveRTFeaturizer,
    LanguageModelFeaturizer,
    # classifiers
    SklearnIntentClassifier,
    MitieIntentClassifier,
    KeywordIntentClassifier,
    DIETClassifier,
    EmbeddingIntentClassifier,
    # selectors
    ResponseSelector,
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
    "intent_classifier_tensorflow_embedding": "EmbeddingIntentClassifier",
}

# To simplify usage, there are a couple of model templates, that already add
# necessary components in the right order. They also implement
# the preexisting `backends`.
registered_pipeline_templates = {
    "pretrained_embeddings_spacy": [
        {"name": "SpacyNLP"},
        {"name": "SpacyTokenizer"},
        {"name": "SpacyFeaturizer"},
        {"name": "RegexFeaturizer"},
        {"name": "CRFEntityExtractor"},
        {"name": "EntitySynonymMapper"},
        {"name": "SklearnIntentClassifier"},
    ],
    "keyword": [{"name": "KeywordIntentClassifier"}],
    "supervised_embeddings": [
        {"name": "WhitespaceTokenizer"},
        {"name": "RegexFeaturizer"},
        {"name": "CRFEntityExtractor"},
        {"name": "EntitySynonymMapper"},
        {"name": "CountVectorsFeaturizer"},
        {
            "name": "CountVectorsFeaturizer",
            "analyzer": "char_wb",
            "min_ngram": 1,
            "max_ngram": 4,
        },
        {"name": "EmbeddingIntentClassifier"},
    ],
    "pretrained_embeddings_convert": [
        {"name": "ConveRTTokenizer"},
        {"name": "ConveRTFeaturizer"},
        {"name": "EmbeddingIntentClassifier"},
    ],
}


def pipeline_template(s: Text) -> Optional[List[Dict[Text, Any]]]:
    import copy

    # do a deepcopy to avoid changing the template configurations
    return copy.deepcopy(registered_pipeline_templates.get(s))


def get_component_class(component_name: Text) -> Type["Component"]:
    """Resolve component name to a registered components class."""

    if component_name not in registered_components:
        if component_name not in old_style_names:
            try:
                return class_from_module_path(component_name)

            except AttributeError:
                # when component_name is a path to a class but the path does not contain
                # that class
                module_name, _, class_name = component_name.rpartition(".")
                raise Exception(
                    f"Failed to find class '{class_name}' in module '{module_name}'.\n"
                )
            except ImportError as e:
                # when component_name is a path to a class but that path is invalid or
                # when component_name is a class name and not part of old_style_names

                is_path = "." in component_name

                if is_path:
                    module_name, _, _ = component_name.rpartition(".")
                    exception_message = f"Failed to find module '{module_name}'. \n{e}"
                else:
                    exception_message = (
                        f"Cannot find class '{component_name}' from global namespace. "
                        f"Please check that there is no typo in the class "
                        f"name and that you have imported the class into the global "
                        f"namespace."
                    )

                raise ModuleNotFoundError(exception_message)
        else:
            # DEPRECATED ensures compatibility, remove in future versions
            raise_warning(
                f"Your nlu config file "
                f"contains old style component name `{component_name}`, "
                f"you should change it to its new class name: "
                f"`{old_style_names[component_name]}`.",
                FutureWarning,
                docs=DOCS_URL_COMPONENTS,
            )
            component_name = old_style_names[component_name]

    return registered_components[component_name]


def load_component_by_meta(
    component_meta: Dict[Text, Any],
    model_dir: Text,
    metadata: Metadata,
    cached_component: Optional["Component"],
    **kwargs: Any,
) -> Optional["Component"]:
    """Resolves a component and calls its load method.

    Inits it based on a previously persisted model.
    """

    # try to get class name first, else create by name
    component_name = component_meta.get("class", component_meta["name"])
    component_class = get_component_class(component_name)
    return component_class.load(
        component_meta, model_dir, metadata, cached_component, **kwargs
    )


def create_component_by_config(
    component_config: Dict[Text, Any], config: "RasaNLUModelConfig"
) -> Optional["Component"]:
    """Resolves a component and calls it's create method.

    Inits it based on a previously persisted model.
    """

    # try to get class name first, else create by name
    component_name = component_config.get("class", component_config["name"])
    component_class = get_component_class(component_name)
    return component_class.create(component_config, config)
