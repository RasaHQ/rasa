TEXT = "text"

RESPONSE_KEY_ATTRIBUTE = "response_key"

INTENT = "intent"

RESPONSE = "response"

ENTITIES = "entities"
BILOU_ENTITIES = "bilou_entities"
BILOU_ENTITIES_ROLE = "bilou_entities_role"
BILOU_ENTITIES_GROUP = "bilou_entities_group"
NO_ENTITY_TAG = "O"

ENTITY_ATTRIBUTE_TYPE = "entity"
ENTITY_ATTRIBUTE_GROUP = "group"
ENTITY_ATTRIBUTE_ROLE = "role"
ENTITY_ATTRIBUTE_VALUE = "value"
ENTITY_ATTRIBUTE_TEXT = "text"
ENTITY_ATTRIBUTE_START = "start"
ENTITY_ATTRIBUTE_END = "end"
ENTITY_ATTRIBUTE_CONFIDENCE = "confidence"
ENTITY_ATTRIBUTE_CONFIDENCE_TYPE = (
    f"{ENTITY_ATTRIBUTE_CONFIDENCE}_{ENTITY_ATTRIBUTE_TYPE}"
)
ENTITY_ATTRIBUTE_CONFIDENCE_GROUP = (
    f"{ENTITY_ATTRIBUTE_CONFIDENCE}_{ENTITY_ATTRIBUTE_GROUP}"
)
ENTITY_ATTRIBUTE_CONFIDENCE_ROLE = (
    f"{ENTITY_ATTRIBUTE_CONFIDENCE}_{ENTITY_ATTRIBUTE_ROLE}"
)

EXTRACTOR = "extractor"

PRETRAINED_EXTRACTORS = {"DucklingHTTPExtractor", "SpacyEntityExtractor"}
TRAINABLE_EXTRACTORS = {"MitieEntityExtractor", "CRFEntityExtractor", "DIETClassifier"}

CLS_TOKEN = "__CLS__"
POSITION_OF_CLS_TOKEN = -1

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
