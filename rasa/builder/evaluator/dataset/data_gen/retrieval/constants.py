# Constants for building documentation index
SITE_BASE_URL = "https://rasa.com/docs"
DOCS_SUBDIR = "docs"
DOCS_EXCLUDED_DIRS = {"snippets", "archive"}

# Langfuse prompt references for the retrieval labeling pipeline
LABELING_CONFIRM_SYSTEM_PROMPT = "data_generation_prompts/retrieval_labeling_confirm"
LABELING_CONFIRM_USER_PROMPT = "data_generation_prompts/retrieval_labeling_confirm_user"
LABELING_SHORTLIST_SYSTEM_PROMPT = (
    "data_generation_prompts/retrieval_labeling_shortlist"
)
LABELING_SHORTLIST_USER_PROMPT = (
    "data_generation_prompts/retrieval_labeling_shortlist_user"
)
