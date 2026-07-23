from typing import Any, List, Type

from rasa.nlu.classifiers.fallback_classifier import FallbackClassifier
from rasa.nlu.classifiers.keyword_intent_classifier import KeywordIntentClassifier
from rasa.nlu.extractors.duckling_entity_extractor import DucklingEntityExtractor
from rasa.nlu.extractors.entity_synonyms import EntitySynonymMapper
from rasa.nlu.extractors.regex_entity_extractor import RegexEntityExtractor
from rasa.nlu.featurizers.sparse_featurizer.lexical_syntactic_featurizer import (
    LexicalSyntacticFeaturizer,
)
from rasa.nlu.featurizers.sparse_featurizer.count_vectors_featurizer import (
    CountVectorsFeaturizer,
)
from rasa.nlu.featurizers.sparse_featurizer.regex_featurizer import RegexFeaturizer
from rasa.nlu.tokenizers.whitespace_tokenizer import WhitespaceTokenizer
from rasa.core.policies.memoization import (
    AugmentedMemoizationPolicy,
    MemoizationPolicy,
)
from rasa.core.policies.rule_policy import RulePolicy
from rasa.shared.utils.common import conditional_import

# Components that depend on optional extras (TensorFlow / skops / spaCy / MITIE /
# jieba / scikit-learn-crfsuite). Each entry is ``(module_path, class_name)``. They
# are imported lazily via ``conditional_import`` so that a rule-based install without
# these extras (and on interpreters where TensorFlow has no wheels, e.g. Python 3.13)
# never imports them. A component that cannot be imported is simply not registered,
# and referencing it in a config raises a helpful ``MissingDependencyException`` from
# ``DefaultV1Recipe._from_registry``.
_CONDITIONAL_COMPONENTS = [
    # TensorFlow-backed components
    ("rasa.nlu.classifiers.diet_classifier", "DIETClassifier"),
    ("rasa.nlu.selectors.response_selector", "ResponseSelector"),
    ("rasa.nlu.featurizers.dense_featurizer.convert_featurizer", "ConveRTFeaturizer"),
    ("rasa.nlu.featurizers.dense_featurizer.lm_featurizer", "LanguageModelFeaturizer"),
    ("rasa.core.policies.ted_policy", "TEDPolicy"),
    ("rasa.core.policies.unexpected_intent_policy", "UnexpecTEDIntentPolicy"),
    # skops / scikit-learn-crfsuite backed components
    (
        "rasa.nlu.classifiers.logistic_regression_classifier",
        "LogisticRegressionClassifier",
    ),
    ("rasa.nlu.classifiers.sklearn_intent_classifier", "SklearnIntentClassifier"),
    ("rasa.nlu.extractors.crf_entity_extractor", "CRFEntityExtractor"),
    # spaCy backed components
    ("rasa.nlu.featurizers.dense_featurizer.spacy_featurizer", "SpacyFeaturizer"),
    ("rasa.nlu.tokenizers.spacy_tokenizer", "SpacyTokenizer"),
    ("rasa.nlu.extractors.spacy_entity_extractor", "SpacyEntityExtractor"),
    ("rasa.nlu.utils.spacy_utils", "SpacyNLP"),
    # MITIE backed components
    ("rasa.nlu.featurizers.dense_featurizer.mitie_featurizer", "MitieFeaturizer"),
    ("rasa.nlu.tokenizers.mitie_tokenizer", "MitieTokenizer"),
    ("rasa.nlu.classifiers.mitie_intent_classifier", "MitieIntentClassifier"),
    ("rasa.nlu.extractors.mitie_entity_extractor", "MitieEntityExtractor"),
    ("rasa.nlu.utils.mitie_utils", "MitieNLP"),
    # jieba backed components
    ("rasa.nlu.tokenizers.jieba_tokenizer", "JiebaTokenizer"),
]


def _build_default_components() -> List[Type[Any]]:
    """Build the list of default components, deferring optional-dependency imports.

    Importing a component module runs its ``@DefaultV1Recipe.register`` decorator, so
    building this list is also what registers the components with the recipe. Base
    components have no optional dependencies and are always available; the rest are
    imported conditionally and only added (and registered) when their extra is
    installed.
    """
    components: List[Type[Any]] = [
        # Message Classifiers
        FallbackClassifier,
        KeywordIntentClassifier,
        # Message Entity Extractors
        DucklingEntityExtractor,
        EntitySynonymMapper,
        RegexEntityExtractor,
        # Message Featurizers
        LexicalSyntacticFeaturizer,
        CountVectorsFeaturizer,
        RegexFeaturizer,
        # Tokenizers
        WhitespaceTokenizer,
        # Dialogue Management Policies
        RulePolicy,
        MemoizationPolicy,
        AugmentedMemoizationPolicy,
    ]

    for module_name, class_name in _CONDITIONAL_COMPONENTS:
        component, available = conditional_import(module_name, class_name)
        if available:
            components.append(component)

    return components


# Built on first import of this module. `DefaultV1Recipe._from_registry` imports this
# lazily (inside the method), so no optional dependency is imported at `import rasa`
# time.
DEFAULT_COMPONENTS = _build_default_components()
