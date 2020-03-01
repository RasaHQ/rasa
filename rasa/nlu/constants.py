TEXT = "text"

RESPONSE_KEY_ATTRIBUTE = "response_key"

INTENT = "intent"

RESPONSE = "response"

ENTITIES = "entities"
BILOU_ENTITIES = "bilou_entities"
NO_ENTITY_TAG = "O"

EXTRACTOR = "extractor"

PRETRAINED_EXTRACTORS = {"DucklingHTTPExtractor", "SpacyEntityExtractor"}

CLS_TOKEN = "__CLS__"

MESSAGE_ATTRIBUTES = [TEXT, INTENT, RESPONSE]

TOKENS_NAMES = {TEXT: "tokens", INTENT: "intent_tokens", RESPONSE: "response_tokens"}

SPARSE_FEATURE_NAMES = {
    TEXT: "text_sparse_features",
    INTENT: "intent_sparse_features",
    RESPONSE: "response_sparse_features",
}

DENSE_FEATURE_NAMES = {
    TEXT: "text_dense_features",
    INTENT: "intent_dense_features",
    RESPONSE: "response_dense_features",
}

LANGUAGE_MODEL_DOCS = {
    TEXT: "text_language_model_doc",
    RESPONSE: "response_language_model_doc",
}

TOKEN_IDS = "token_ids"
TOKENS = "tokens"
SEQUENCE_FEATURES = "sequence_features"
SENTENCE_FEATURES = "sentence_features"

SPACY_DOCS = {TEXT: "text_spacy_doc", RESPONSE: "response_spacy_doc"}


DENSE_FEATURIZABLE_ATTRIBUTES = [TEXT, RESPONSE]

RESPONSE_SELECTOR_PROPERTY_NAME = "response_selector"
DEFAULT_OPEN_UTTERANCE_TYPE = "default"
OPEN_UTTERANCE_PREDICTION_KEY = "response"
OPEN_UTTERANCE_RANKING_KEY = "ranking"
RESPONSE_IDENTIFIER_DELIMITER = "/"
