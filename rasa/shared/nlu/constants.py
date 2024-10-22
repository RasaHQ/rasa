TEXT = "text"
TEXT_TOKENS = "text_tokens"
INTENT = "intent"
COMMANDS = "commands"
LLM_COMMANDS = "llm_commands"  # needed for fine-tuning
LLM_PROMPT = "llm_prompt"  # needed for fine-tuning
FLOWS_FROM_SEMANTIC_SEARCH = "flows_from_semantic_search"
FLOWS_IN_PROMPT = "flows_in_prompt"
NOT_INTENT = "not_intent"
RESPONSE = "response"
RESPONSE_SELECTOR = "response_selector"
INTENT_RESPONSE_KEY = "intent_response_key"
ACTION_TEXT = "action_text"
ACTION_NAME = "action_name"
INTENT_NAME_KEY = "name"
FULL_RETRIEVAL_INTENT_NAME_KEY = "full_retrieval_intent_name"
METADATA = "metadata"
METADATA_INTENT = "intent"
METADATA_EXAMPLE = "example"
METADATA_MODEL_ID = "model_id"
INTENT_RANKING_KEY = "intent_ranking"
PREDICTED_CONFIDENCE_KEY = "confidence"

RESPONSE_IDENTIFIER_DELIMITER = "/"

FEATURE_TYPE_SENTENCE = "sentence"
FEATURE_TYPE_SEQUENCE = "sequence"
VALID_FEATURE_TYPES = [FEATURE_TYPE_SEQUENCE, FEATURE_TYPE_SENTENCE]

EXTRACTOR = "extractor"
PRETRAINED_EXTRACTORS = {"DucklingEntityExtractor", "SpacyEntityExtractor"}
TRAINABLE_EXTRACTORS = {"MitieEntityExtractor", "CRFEntityExtractor", "DIETClassifier"}

ENTITIES = "entities"
ENTITY_TAGS = "entity_tags"
ENTITY_ATTRIBUTE_TYPE = "entity"
ENTITY_ATTRIBUTE_GROUP = "group"
ENTITY_ATTRIBUTE_ROLE = "role"
ENTITY_ATTRIBUTE_VALUE = "value"
ENTITY_ATTRIBUTE_START = "start"
ENTITY_ATTRIBUTE_END = "end"
ENTITY_ATTRIBUTE_TEXT = "text"
ENTITY_ATTRIBUTE_CONFIDENCE = "confidence"
NO_ENTITY_TAG = "O"
SPLIT_ENTITIES_BY_COMMA = "split_entities_by_comma"
SPLIT_ENTITIES_BY_COMMA_DEFAULT_VALUE = True
SINGLE_ENTITY_ALLOWED_INTERLEAVING_CHARSET = {".", ",", " ", ";"}

SET_SLOT_COMMAND = "set slot"
