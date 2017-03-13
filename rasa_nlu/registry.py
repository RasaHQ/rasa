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

component_classes = [
    SpacyNLP, SpacyEntityExtractor, SklearnIntentClassifier, SpacyFeaturizer,
    MitieNLP, MitieEntityExtractor, MitieIntentClassifier, MitieFeaturizer, MitieTokenizer,
    KeywordIntentClassifier, EntitySynonymMapper, NGramFeaturizer]


registered_components = {component.name: component for component in component_classes}


registered_model_templates = {
    "spacy_sklearn": [
        "init_spacy",
        "ner_spacy",
        "ner_synonyms",
        "intent_featurizer_spacy",
        "intent_sklearn",
    ],
    "mitie": [
        "init_mitie",
        "tokenizer_mitie",
        "ner_mitie",
        "ner_synonyms",
        "intent_featurizer_mitie",
        "intent_mitie",
    ],
    "mitie_sklearn": [
        "init_mitie",
        "tokenizer_mitie",
        "ner_mitie",
        "ner_synonyms",
        "intent_featurizer_mitie",
        "intent_sklearn",
    ],
    "keyword": [
        "intent_keyword",
    ]
}
