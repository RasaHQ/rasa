import importlib_resources

from rasa.constants import PACKAGE_NAME

INKEEP_API_KEY_ENV_VAR = "INKEEP_API_KEY"

INKEEP_RAG_RESPONSE_SCHEMA_PATH = str(
    importlib_resources.files(PACKAGE_NAME)
    .joinpath("builder")
    .joinpath("document_retrieval")
    .joinpath("inkeep-rag-response-schema.json")
)


INKEEP_DOCUMENT_RETRIEVAL_MODEL = "inkeep-rag"
