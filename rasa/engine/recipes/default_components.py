from rasa.core.policies.enterprise_search_policy import EnterpriseSearchPolicy
from rasa.core.policies.flow_policy import FlowPolicy
from rasa.core.policies.intentless_policy import IntentlessPolicy
from rasa.core.policies.memoization import AugmentedMemoizationPolicy, MemoizationPolicy
from rasa.core.policies.rule_policy import RulePolicy
from rasa.dialogue_understanding.coexistence.intent_based_router import (
    IntentBasedRouter,
)
from rasa.dialogue_understanding.coexistence.llm_based_router import LLMBasedRouter
from rasa.dialogue_understanding.generator import (
    LLMCommandGenerator,
)
from rasa.dialogue_understanding.generator.nlu_command_adapter import NLUCommandAdapter
from rasa.nlu.classifiers.fallback_classifier import FallbackClassifier
from rasa.nlu.classifiers.keyword_intent_classifier import KeywordIntentClassifier
from rasa.nlu.extractors.duckling_entity_extractor import DucklingEntityExtractor
from rasa.nlu.extractors.entity_synonyms import EntitySynonymMapper
from rasa.nlu.extractors.regex_entity_extractor import RegexEntityExtractor
from rasa.nlu.featurizers.sparse_featurizer.count_vectors_featurizer import (
    CountVectorsFeaturizer,
)
from rasa.nlu.featurizers.sparse_featurizer.regex_featurizer import RegexFeaturizer
from rasa.nlu.tokenizers.whitespace_tokenizer import WhitespaceTokenizer
from rasa.shared.utils.common import conditional_import

# Conditional imports for components with external dependencies

# components dependent on tensorflow
TEDPolicy, TED_POLICY_AVAILABLE = conditional_import(
    "rasa.core.policies.ted_policy", "TEDPolicy", check_installation_setup=True
)
UnexpecTEDIntentPolicy, UNEXPECTED_INTENT_POLICY_AVAILABLE = conditional_import(
    "rasa.core.policies.unexpected_intent_policy",
    "UnexpecTEDIntentPolicy",
    check_installation_setup=True,
)
DIETClassifier, DIET_CLASSIFIER_AVAILABLE = conditional_import(
    "rasa.nlu.classifiers.diet_classifier",
    "DIETClassifier",
    check_installation_setup=True,
)
ConveRTFeaturizer, CONVERT_FEATURIZER_AVAILABLE = conditional_import(
    "rasa.nlu.featurizers.dense_featurizer.convert_featurizer",
    "ConveRTFeaturizer",
    check_installation_setup=True,
)
LanguageModelFeaturizer, LANGUAGE_MODEL_FEATURIZER_AVAILABLE = conditional_import(
    "rasa.nlu.featurizers.dense_featurizer.lm_featurizer",
    "LanguageModelFeaturizer",
    check_installation_setup=True,
)
ResponseSelector, RESPONSE_SELECTOR_AVAILABLE = conditional_import(
    "rasa.nlu.selectors.response_selector",
    "ResponseSelector",
    check_installation_setup=True,
)

# components dependent on skops
LogisticRegressionClassifier, LOGISTIC_REGRESSION_CLASSIFIER_AVAILABLE = (
    conditional_import(
        "rasa.nlu.classifiers.logistic_regression_classifier",
        "LogisticRegressionClassifier",
    )
)
SklearnIntentClassifier, SKLEARN_INTENT_CLASSIFIER_AVAILABLE = conditional_import(
    "rasa.nlu.classifiers.sklearn_intent_classifier", "SklearnIntentClassifier"
)

# components dependent on spacy
LexicalSyntacticFeaturizer, LEXICAL_SYNTACTIC_FEATURIZER_AVAILABLE = conditional_import(
    "rasa.nlu.featurizers.sparse_featurizer.lexical_syntactic_featurizer",
    "LexicalSyntacticFeaturizer",
)
SpacyFeaturizer, SPACY_FEATURIZER_AVAILABLE = conditional_import(
    "rasa.nlu.featurizers.dense_featurizer.spacy_featurizer", "SpacyFeaturizer"
)
SpacyTokenizer, SPACY_TOKENIZER_AVAILABLE = conditional_import(
    "rasa.nlu.tokenizers.spacy_tokenizer", "SpacyTokenizer"
)
SpacyEntityExtractor, SPACY_ENTITY_EXTRACTOR_AVAILABLE = conditional_import(
    "rasa.nlu.extractors.spacy_entity_extractor", "SpacyEntityExtractor"
)
SpacyNLP, SPACY_NLP_AVAILABLE = conditional_import(
    "rasa.nlu.utils.spacy_utils", "SpacyNLP"
)

# components dependent on sklearn_crfsuite
CRFEntityExtractor, CRF_ENTITY_EXTRACTOR_AVAILABLE = conditional_import(
    "rasa.nlu.extractors.crf_entity_extractor", "CRFEntityExtractor"
)

# components dependent on mitie
MitieFeaturizer, MITIE_FEATURIZER_AVAILABLE = conditional_import(
    "rasa.nlu.featurizers.dense_featurizer.mitie_featurizer", "MitieFeaturizer"
)
MitieTokenizer, MITIE_TOKENIZER_AVAILABLE = conditional_import(
    "rasa.nlu.tokenizers.mitie_tokenizer", "MitieTokenizer"
)
MitieIntentClassifier, MITIE_INTENT_CLASSIFIER_AVAILABLE = conditional_import(
    "rasa.nlu.classifiers.mitie_intent_classifier", "MitieIntentClassifier"
)
MitieEntityExtractor, MITIE_ENTITY_EXTRACTOR_AVAILABLE = conditional_import(
    "rasa.nlu.extractors.mitie_entity_extractor", "MitieEntityExtractor"
)
MitieNLP, MITIE_NLP_AVAILABLE = conditional_import(
    "rasa.nlu.utils.mitie_utils", "MitieNLP"
)

# components dependent on jieba
JiebaTokenizer, JIEBA_TOKENIZER_AVAILABLE = conditional_import(
    "rasa.nlu.tokenizers.jieba_tokenizer", "JiebaTokenizer"
)

# Base components that are always available (no external dependencies)
DEFAULT_COMPONENTS = [
    # Classifiers
    FallbackClassifier,
    KeywordIntentClassifier,
    NLUCommandAdapter,
    LLMCommandGenerator,
    LLMBasedRouter,
    IntentBasedRouter,
    # Entity Extractors
    DucklingEntityExtractor,
    EntitySynonymMapper,
    RegexEntityExtractor,
    # Featurizers
    CountVectorsFeaturizer,
    RegexFeaturizer,
    # Tokenizers
    WhitespaceTokenizer,
    # Language Model Providers
    # Policies
    RulePolicy,
    MemoizationPolicy,
    AugmentedMemoizationPolicy,
    FlowPolicy,
    EnterpriseSearchPolicy,
    IntentlessPolicy,
]

# Conditionally add components based on dependencies

# components dependent on tensorflow
if DIET_CLASSIFIER_AVAILABLE:
    DEFAULT_COMPONENTS.append(DIETClassifier)
if CONVERT_FEATURIZER_AVAILABLE:
    DEFAULT_COMPONENTS.append(ConveRTFeaturizer)
if LANGUAGE_MODEL_FEATURIZER_AVAILABLE:
    DEFAULT_COMPONENTS.append(LanguageModelFeaturizer)
if RESPONSE_SELECTOR_AVAILABLE:
    DEFAULT_COMPONENTS.append(ResponseSelector)
if TED_POLICY_AVAILABLE:
    DEFAULT_COMPONENTS.append(TEDPolicy)
if UNEXPECTED_INTENT_POLICY_AVAILABLE:
    DEFAULT_COMPONENTS.append(UnexpecTEDIntentPolicy)

# components dependent on skops
if LOGISTIC_REGRESSION_CLASSIFIER_AVAILABLE:
    DEFAULT_COMPONENTS.append(LogisticRegressionClassifier)
if SKLEARN_INTENT_CLASSIFIER_AVAILABLE:
    DEFAULT_COMPONENTS.append(SklearnIntentClassifier)

# components dependent on spacy
if LEXICAL_SYNTACTIC_FEATURIZER_AVAILABLE:
    DEFAULT_COMPONENTS.append(LexicalSyntacticFeaturizer)
if SPACY_FEATURIZER_AVAILABLE:
    DEFAULT_COMPONENTS.append(SpacyFeaturizer)
if SPACY_TOKENIZER_AVAILABLE:
    DEFAULT_COMPONENTS.append(SpacyTokenizer)
if SPACY_ENTITY_EXTRACTOR_AVAILABLE:
    DEFAULT_COMPONENTS.append(SpacyEntityExtractor)
if SPACY_NLP_AVAILABLE:
    DEFAULT_COMPONENTS.append(SpacyNLP)

# components dependent on mitie
if MITIE_FEATURIZER_AVAILABLE:
    DEFAULT_COMPONENTS.append(MitieFeaturizer)
if MITIE_TOKENIZER_AVAILABLE:
    DEFAULT_COMPONENTS.append(MitieTokenizer)
if MITIE_INTENT_CLASSIFIER_AVAILABLE:
    DEFAULT_COMPONENTS.append(MitieIntentClassifier)
if MITIE_ENTITY_EXTRACTOR_AVAILABLE:
    DEFAULT_COMPONENTS.append(MitieEntityExtractor)
if MITIE_NLP_AVAILABLE:
    DEFAULT_COMPONENTS.append(MitieNLP)

# components dependent on jieba
if JIEBA_TOKENIZER_AVAILABLE:
    DEFAULT_COMPONENTS.append(JiebaTokenizer)

# components dependent on sklearn_crfsuite
if CRF_ENTITY_EXTRACTOR_AVAILABLE:
    DEFAULT_COMPONENTS.append(CRFEntityExtractor)
