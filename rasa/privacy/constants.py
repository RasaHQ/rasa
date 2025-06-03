PRIVACY_CONFIG_SCHEMA = "privacy/privacy_config_schema.json"
REDACTION_CHAR_KEY = "redaction_char"
KEEP_LEFT_KEY = "keep_left"
KEEP_RIGHT_KEY = "keep_right"
DELETION_KEY = "deletion"
ANONYMIZATION_KEY = "anonymization"
TRACKER_STORE_SETTINGS = "tracker_store_settings"
SLOT_KEY = "slot"
TEXT_KEY = "text"
ENTITIES_KEY = "entities"
VALUE_KEY = "value"
ENTITY_LABEL_KEY = "label"

USER_CHAT_INACTIVITY_IN_MINUTES_ENV_VAR_NAME = "USER_CHAT_INACTIVITY_IN_MINUTES"
GLINER_MODEL_PATH_ENV_VAR_NAME = "GLINER_MODEL_PATH"
HUGGINGFACE_CACHE_DIR_ENV_VAR_NAME = "HUGGINGFACE_HUB_CACHE_DIR"

DEFAULT_PII_MODEL = "urchade/gliner_multi_pii-v1"
GLINER_LABELS = [
    "person",
    "organization",
    "company",
    "phone number",
    "address",
    "full address",
    "postcode",
    "zip code",
    "passport number",
    "email",
    "credit card number",
    "social security number",
    "health insurance id number",
    "date of birth",
    "mobile phone number",
    "bank account number",
    "medication",
    "cpf",
    "driver's license number",
    "tax identification number",
    "medical condition",
    "identity card number",
    "national id number",
    "ip address",
    "email address",
    "iban",
    "credit card expiration date",
    "username",
    "health insurance number",
    "registration number",
    "student id number",
    "insurance number",
    "membership number",
    "booking number",
    "landline phone number",
    "blood type",
    "cvv",
    "reservation number",
    "digital signature",
    "social media handle",
    "license plate number",
    "cnpj",
    "postal code",
    "passport_number",
    "serial number",
    "vehicle registration number",
    "fax number",
    "visa number",
    "insurance company",
    "identity document number",
    "transaction number",
    "national health insurance number",
    "cvc",
    "birth certificate number",
    "train ticket number",
    "passport expiration date",
    "social_security_number",
    "personally identifiable information",
    "banking routing number",
    "sort code",
    "routing number",
    "tax number",
    "swift code",
]
