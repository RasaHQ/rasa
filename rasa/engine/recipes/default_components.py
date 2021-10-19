from rasa.nlu.classifiers.diet_classifier import DIETClassifierGraphComponent
from rasa.nlu.classifiers.fallback_classifier import FallbackClassifierGraphComponent
from rasa.nlu.classifiers.keyword_intent_classifier import (
    KeywordIntentClassifierGraphComponent,
)
from rasa.nlu.classifiers.mitie_intent_classifier import (
    MitieIntentClassifierGraphComponent,
)
from rasa.nlu.classifiers.sklearn_intent_classifier import (
    SklearnIntentClassifierGraphComponent,
)
from rasa.nlu.extractors.crf_entity_extractor import CRFEntityExtractorGraphComponent
from rasa.nlu.extractors.duckling_entity_extractor import (
    DucklingEntityExtractorGraphComponent,
)
from rasa.nlu.extractors.entity_synonyms import EntitySynonymMapperGraphComponent
from rasa.nlu.extractors.mitie_entity_extractor import (
    MitieEntityExtractorGraphComponent,
)
from rasa.nlu.extractors.spacy_entity_extractor import (
    SpacyEntityExtractorGraphComponent,
)
from rasa.nlu.extractors.regex_entity_extractor import (
    RegexEntityExtractorGraphComponent,
)
from rasa.nlu.featurizers.sparse_featurizer.lexical_syntactic_featurizer import (
    LexicalSyntacticFeaturizerGraphComponent,
)
from rasa.nlu.featurizers.dense_featurizer.convert_featurizer import (
    ConveRTFeaturizerGraphComponent,
)
from rasa.nlu.featurizers.dense_featurizer.mitie_featurizer import (
    MitieFeaturizerGraphComponent,
)
from rasa.nlu.featurizers.dense_featurizer.spacy_featurizer import (
    SpacyFeaturizerGraphComponent,
)
from rasa.nlu.featurizers.sparse_featurizer.count_vectors_featurizer import (
    CountVectorsFeaturizerGraphComponent,
)
from rasa.nlu.featurizers.dense_featurizer.lm_featurizer import (
    LanguageModelFeaturizerGraphComponent,
)
from rasa.nlu.featurizers.sparse_featurizer.regex_featurizer import (
    RegexFeaturizerGraphComponent,
)
from rasa.nlu.selectors.response_selector import ResponseSelectorGraphComponent
from rasa.nlu.tokenizers.jieba_tokenizer import JiebaTokenizerGraphComponent
from rasa.nlu.tokenizers.mitie_tokenizer import MitieTokenizerGraphComponent
from rasa.nlu.tokenizers.spacy_tokenizer import SpacyTokenizerGraphComponent
from rasa.nlu.tokenizers.whitespace_tokenizer import WhitespaceTokenizerGraphComponent
from rasa.nlu.utils.mitie_utils import MitieNLPGraphComponent
from rasa.nlu.utils.spacy_utils import SpacyNLPGraphComponent

from rasa.core.policies.ted_policy import TEDPolicyGraphComponent
from rasa.core.policies.memoization import (
    MemoizationPolicyGraphComponent,
    AugmentedMemoizationPolicyGraphComponent,
)
from rasa.core.policies.rule_policy import RulePolicyGraphComponent
from rasa.core.policies.unexpected_intent_policy import (
    UnexpecTEDIntentPolicyGraphComponent,
)

DEFAULT_COMPONENTS = [
    # Message Classifiers
    DIETClassifierGraphComponent,
    FallbackClassifierGraphComponent,
    KeywordIntentClassifierGraphComponent,
    MitieIntentClassifierGraphComponent,
    SklearnIntentClassifierGraphComponent,
    # Response Selectors
    ResponseSelectorGraphComponent,
    # Message Entity Extractors
    CRFEntityExtractorGraphComponent,
    DucklingEntityExtractorGraphComponent,
    EntitySynonymMapperGraphComponent,
    MitieEntityExtractorGraphComponent,
    SpacyEntityExtractorGraphComponent,
    RegexEntityExtractorGraphComponent,
    # Message Feauturizers
    LexicalSyntacticFeaturizerGraphComponent,
    ConveRTFeaturizerGraphComponent,
    MitieFeaturizerGraphComponent,
    SpacyFeaturizerGraphComponent,
    CountVectorsFeaturizerGraphComponent,
    LanguageModelFeaturizerGraphComponent,
    RegexFeaturizerGraphComponent,
    # Tokenizers
    JiebaTokenizerGraphComponent,
    MitieTokenizerGraphComponent,
    SpacyTokenizerGraphComponent,
    WhitespaceTokenizerGraphComponent,
    # Language Model Providers
    MitieNLPGraphComponent,
    SpacyNLPGraphComponent,
    # Dialogue Management Policies
    TEDPolicyGraphComponent,
    UnexpecTEDIntentPolicyGraphComponent,
    RulePolicyGraphComponent,
    MemoizationPolicyGraphComponent,
    AugmentedMemoizationPolicyGraphComponent,
]
