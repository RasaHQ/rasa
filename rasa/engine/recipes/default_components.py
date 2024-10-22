from rasa.core.policies.enterprise_search_policy import EnterpriseSearchPolicy
from rasa.core.policies.flow_policy import FlowPolicy
from rasa.dialogue_understanding.coexistence.intent_based_router import (
    IntentBasedRouter,
)
from rasa.dialogue_understanding.coexistence.llm_based_router import LLMBasedRouter
from rasa.core.policies.intentless_policy import IntentlessPolicy
from rasa.dialogue_understanding.generator.nlu_command_adapter import NLUCommandAdapter
from rasa.nlu.classifiers.diet_classifier import DIETClassifier
from rasa.nlu.classifiers.fallback_classifier import FallbackClassifier
from rasa.nlu.classifiers.keyword_intent_classifier import KeywordIntentClassifier
from rasa.dialogue_understanding.generator import (
    LLMCommandGenerator,
)
from rasa.nlu.classifiers.logistic_regression_classifier import (
    LogisticRegressionClassifier,
)
from rasa.nlu.classifiers.mitie_intent_classifier import MitieIntentClassifier
from rasa.nlu.classifiers.sklearn_intent_classifier import SklearnIntentClassifier
from rasa.nlu.extractors.crf_entity_extractor import CRFEntityExtractor
from rasa.nlu.extractors.duckling_entity_extractor import DucklingEntityExtractor
from rasa.nlu.extractors.entity_synonyms import EntitySynonymMapper
from rasa.nlu.extractors.mitie_entity_extractor import MitieEntityExtractor
from rasa.nlu.extractors.spacy_entity_extractor import SpacyEntityExtractor
from rasa.nlu.extractors.regex_entity_extractor import RegexEntityExtractor
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
from rasa.nlu.selectors.response_selector import ResponseSelector
from rasa.nlu.tokenizers.jieba_tokenizer import JiebaTokenizer
from rasa.nlu.tokenizers.mitie_tokenizer import MitieTokenizer
from rasa.nlu.tokenizers.spacy_tokenizer import SpacyTokenizer
from rasa.nlu.tokenizers.whitespace_tokenizer import WhitespaceTokenizer
from rasa.nlu.utils.mitie_utils import MitieNLP
from rasa.nlu.utils.spacy_utils import SpacyNLP


from rasa.core.policies.ted_policy import TEDPolicy
from rasa.core.policies.memoization import MemoizationPolicy, AugmentedMemoizationPolicy
from rasa.core.policies.rule_policy import RulePolicy
from rasa.core.policies.unexpected_intent_policy import UnexpecTEDIntentPolicy

DEFAULT_COMPONENTS = [
    # Message Classifiers
    DIETClassifier,
    FallbackClassifier,
    KeywordIntentClassifier,
    MitieIntentClassifier,
    SklearnIntentClassifier,
    LogisticRegressionClassifier,
    NLUCommandAdapter,
    LLMCommandGenerator,
    LLMBasedRouter,
    IntentBasedRouter,
    # Response Selectors
    ResponseSelector,
    # Message Entity Extractors
    CRFEntityExtractor,
    DucklingEntityExtractor,
    EntitySynonymMapper,
    MitieEntityExtractor,
    SpacyEntityExtractor,
    RegexEntityExtractor,
    # Message Feauturizers
    LexicalSyntacticFeaturizer,
    ConveRTFeaturizer,
    MitieFeaturizer,
    SpacyFeaturizer,
    CountVectorsFeaturizer,
    LanguageModelFeaturizer,
    RegexFeaturizer,
    # Tokenizers
    JiebaTokenizer,
    MitieTokenizer,
    SpacyTokenizer,
    WhitespaceTokenizer,
    # Language Model Providers
    MitieNLP,
    SpacyNLP,
    # Dialogue Management Policies
    TEDPolicy,
    UnexpecTEDIntentPolicy,
    RulePolicy,
    MemoizationPolicy,
    AugmentedMemoizationPolicy,
    FlowPolicy,
    EnterpriseSearchPolicy,
    IntentlessPolicy,
]
