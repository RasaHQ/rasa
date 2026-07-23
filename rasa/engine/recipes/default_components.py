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

# This build targets Python 3.12/3.13, where TensorFlow has no wheels. The
# TensorFlow- and ML-backed components (DIETClassifier, TEDPolicy,
# UnexpecTEDIntentPolicy, ResponseSelector, ConveRTFeaturizer,
# LanguageModelFeaturizer, the sklearn/skops classifiers, and all
# spaCy/MITIE/jieba components) have been removed from this distribution. Only the
# non-ML, rule-based components remain — suitable for assistants that use direct
# intent triggering and rules.
DEFAULT_COMPONENTS = [
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
