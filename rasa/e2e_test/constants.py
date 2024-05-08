SCHEMA_FILE_PATH = "e2e_test/e2e_test_schema.yml"

KEY_FIXTURES = "fixtures"
KEY_USER_INPUT = "user"
KEY_BOT_INPUT = "bot"
KEY_BOT_UTTERED = "utter"
KEY_SLOT_SET = "slot_was_set"
KEY_SLOT_NOT_SET = "slot_was_not_set"
KEY_STEPS = "steps"
KEY_TEST_CASE = "test_case"
KEY_COMMANDS = "commands"

AVAILABLE_COMMANDS = [
    "start_flow",
    "set_slot",
    "correct_slot",
    "cancel_flow",
    "clarify",
    "chitchat",
    "human_handoff",
    "knowledge",
    "skip_question",
    "error",
]
