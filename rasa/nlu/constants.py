import rasa.shared.nlu.constants


BILOU_ENTITIES = "bilou_entities"
BILOU_ENTITIES_ROLE = "bilou_entities_role"
BILOU_ENTITIES_GROUP = "bilou_entities_group"

ENTITY_ATTRIBUTE_TEXT = "text"
ENTITY_ATTRIBUTE_CONFIDENCE = "confidence"
ENTITY_ATTRIBUTE_CONFIDENCE_TYPE = (
    f"{ENTITY_ATTRIBUTE_CONFIDENCE}_{rasa.shared.nlu.constants.ENTITY_ATTRIBUTE_TYPE}"
)
ENTITY_ATTRIBUTE_CONFIDENCE_GROUP = (
    f"{ENTITY_ATTRIBUTE_CONFIDENCE}_{rasa.shared.nlu.constants.ENTITY_ATTRIBUTE_GROUP}"
)
ENTITY_ATTRIBUTE_CONFIDENCE_ROLE = (
    f"{ENTITY_ATTRIBUTE_CONFIDENCE}_{rasa.shared.nlu.constants.ENTITY_ATTRIBUTE_ROLE}"
)

EXTRACTOR = "extractor"

PRETRAINED_EXTRACTORS = {
    "DucklingEntityExtractor",
    "DucklingHTTPExtractor",  # for backwards compatibility when dumping Markdown
    "SpacyEntityExtractor",
}
TRAINABLE_EXTRACTORS = {"MitieEntityExtractor", "CRFEntityExtractor", "DIETClassifier"}

NUMBER_OF_SUB_TOKENS = "number_of_sub_tokens"

MESSAGE_ATTRIBUTES = [
    rasa.shared.nlu.constants.TEXT,
    rasa.shared.nlu.constants.INTENT,
    rasa.shared.nlu.constants.RESPONSE,
    rasa.shared.nlu.constants.ACTION_NAME,
    rasa.shared.nlu.constants.ACTION_TEXT,
    rasa.shared.nlu.constants.INTENT_RESPONSE_KEY,
]
# the dense featurizable attributes are essentially text attributes
DENSE_FEATURIZABLE_ATTRIBUTES = [
    rasa.shared.nlu.constants.TEXT,
    rasa.shared.nlu.constants.RESPONSE,
    rasa.shared.nlu.constants.ACTION_TEXT,
]

LANGUAGE_MODEL_DOCS = {
    rasa.shared.nlu.constants.TEXT: "text_language_model_doc",
    rasa.shared.nlu.constants.RESPONSE: "response_language_model_doc",
    rasa.shared.nlu.constants.ACTION_TEXT: "action_text_model_doc",
}
SPACY_DOCS = {
    rasa.shared.nlu.constants.TEXT: "text_spacy_doc",
    rasa.shared.nlu.constants.RESPONSE: "response_spacy_doc",
    rasa.shared.nlu.constants.ACTION_TEXT: "action_text_spacy_doc",
}

TOKENS_NAMES = {
    rasa.shared.nlu.constants.TEXT: "text_tokens",
    rasa.shared.nlu.constants.INTENT: "intent_tokens",
    rasa.shared.nlu.constants.RESPONSE: "response_tokens",
    rasa.shared.nlu.constants.ACTION_NAME: "action_name_tokens",
    rasa.shared.nlu.constants.ACTION_TEXT: "action_text_tokens",
    rasa.shared.nlu.constants.INTENT_RESPONSE_KEY: "intent_response_key_tokens",
}

SEQUENCE_FEATURES = "sequence_features"
SENTENCE_FEATURES = "sentence_features"

RESPONSE_SELECTOR_PROPERTY_NAME = "response_selector"
RESPONSE_SELECTOR_RETRIEVAL_INTENTS = "all_retrieval_intents"
RESPONSE_SELECTOR_DEFAULT_INTENT = "default"
RESPONSE_SELECTOR_PREDICTION_KEY = "response"
RESPONSE_SELECTOR_RANKING_KEY = "ranking"
RESPONSE_SELECTOR_RESPONSES_KEY = "response_templates"
RESPONSE_SELECTOR_TEMPLATE_NAME_KEY = "template_name"
RESPONSE_IDENTIFIER_DELIMITER = "/"

FEATURIZER_CLASS_ALIAS = "alias"

NO_LENGTH_RESTRICTION = -1
