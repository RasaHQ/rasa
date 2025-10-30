SCHEMA_FILE_PATH = "e2e_test/e2e_test_schema.yml"
E2E_CONFIG_SCHEMA_FILE_PATH = "e2e_test/e2e_config_schema.yml"
TEST_FILE_NAME = "test_file_name"
TEST_CASE_NAME = "test_case_name"
STUB_CUSTOM_ACTION_NAME_SEPARATOR = "::"

KEY_FIXTURES = "fixtures"
KEY_USER_INPUT = "user"
KEY_BOT_INPUT = "bot"
KEY_BOT_UTTERED = "utter"
KEY_SLOT_SET = "slot_was_set"
KEY_SLOT_NOT_SET = "slot_was_not_set"
KEY_STEPS = "steps"
KEY_TEST_CASE = "test_case"
KEY_TEST_CASES = "test_cases"
KEY_METADATA = "metadata"
KEY_ASSERTIONS = "assertions"
KEY_ASSERTION_ORDER_ENABLED = "assertion_order_enabled"
KEY_STUB_CUSTOM_ACTIONS = "stub_custom_actions"
KEY_THRESHOLD = "threshold"
KEY_UTTER_NAME = "utter_name"
KEY_GROUND_TRUTH = "ground_truth"
KEY_UTTER_SOURCE = "utter_source"

KEY_MODEL = "model"
KEY_LLM_JUDGE = "llm_judge"
KEY_LLM_E2E_TEST_CONVERSION = "llm_e2e_test_conversion"

DEFAULT_E2E_INPUT_TESTS_PATH = "tests/e2e_test_cases.yml"
DEFAULT_E2E_OUTPUT_TESTS_PATH = "tests/e2e_results.yml"
DEFAULT_COVERAGE_OUTPUT_PATH = "e2e_coverage_results"
DEFAULT_E2E_FAILED_TESTS_PATH = "tests/e2e_failed_tests.yml"

# Test status
STATUS_PASSED = "passed"
STATUS_FAILED = "failed"

# LLM Judge
LLM_JUDGE_PROMPTS_MODULE = "rasa.e2e_test.llm_judge_prompts"
DEFAULT_GROUNDEDNESS_PROMPT_TEMPLATE_FILE_NAME = "groundedness_prompt_template.jinja2"
DEFAULT_ANSWER_RELEVANCE_PROMPT_TEMPLATE_FILE_NAME = (
    "answer_relevance_prompt_template.jinja2"
)
DEFAULT_E2E_TESTING_MODEL = "gpt-4.1-mini-2025-04-14"
KEY_SCORE = "score"
KEY_JUSTIFICATION = "justification"
KEY_EXTRA_PARAMETERS = "extra_parameters"
