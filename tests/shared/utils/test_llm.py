from typing import Text, Any, Dict, Optional

import pytest
from langchain import OpenAI
from langchain.chat_models import AzureChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from pathlib import Path
from pytest import MonkeyPatch
from rasa.shared.constants import (
    RASA_PATTERN_INTERNAL_ERROR_USER_INPUT_TOO_LONG,
    RASA_PATTERN_INTERNAL_ERROR_USER_INPUT_EMPTY,
)
from rasa.shared.core.domain import Domain
from rasa.shared.core.events import BotUttered, UserUttered, SessionStarted, Restarted
from rasa.shared.core.slots import (
    FloatSlot,
    TextSlot,
    BooleanSlot,
    CategoricalSlot,
    Slot,
)
from rasa.shared.core.trackers import DialogueStateTracker
from rasa.shared.utils.llm import (
    get_prompt_template,
    sanitize_message_for_prompt,
    tracker_as_readable_transcript,
    embedder_factory,
    llm_factory,
    ERROR_PLACEHOLDER,
    allowed_values_for_slot,
)


def test_tracker_as_readable_transcript_handles_empty_tracker():
    tracker = DialogueStateTracker(sender_id="test", slots=[])
    assert tracker_as_readable_transcript(tracker) == ""


def test_tracker_as_readable_transcript_handles_tracker_with_events(domain: Domain):
    tracker = DialogueStateTracker(sender_id="test", slots=domain.slots)
    tracker.update_with_events(
        [
            UserUttered("hello"),
            BotUttered("hi"),
        ],
    )
    assert tracker_as_readable_transcript(tracker) == ("""USER: hello\nAI: hi""")


def test_tracker_as_readable_transcript_handles_session_restart(domain: Domain):
    tracker = DialogueStateTracker(sender_id="test", slots=domain.slots)
    tracker.update_with_events(
        [
            UserUttered("hello"),
            BotUttered("hi"),
            # this should clear the prior conversation from the transcript
            SessionStarted(),
            UserUttered("howdy"),
        ],
    )
    assert tracker_as_readable_transcript(tracker) == ("""USER: howdy""")


def test_tracker_as_readable_transcript_handles_restart(domain: Domain):
    tracker = DialogueStateTracker(sender_id="test", slots=domain.slots)
    tracker.update_with_events(
        [
            UserUttered("hello"),
            BotUttered("hi"),
            # this should clear the prior conversation from the transcript
            Restarted(),
            UserUttered("howdy"),
        ],
    )
    assert tracker_as_readable_transcript(tracker) == ("""USER: howdy""")


def test_tracker_as_readable_transcript_handles_tracker_with_events_and_prefixes(
    domain: Domain,
):
    tracker = DialogueStateTracker(sender_id="test", slots=domain.slots)
    tracker.update_with_events(
        [
            UserUttered("hello"),
            BotUttered("hi"),
        ],
        domain,
    )
    assert tracker_as_readable_transcript(
        tracker, human_prefix="FOO", ai_prefix="BAR"
    ) == ("""FOO: hello\nBAR: hi""")


def test_tracker_as_readable_transcript_handles_tracker_with_events_and_max_turns(
    domain: Domain,
):
    tracker = DialogueStateTracker(sender_id="test", slots=domain.slots)
    tracker.update_with_events(
        [
            UserUttered("hello"),
            BotUttered("hi"),
        ],
        domain,
    )
    assert tracker_as_readable_transcript(tracker, max_turns=1) == ("""AI: hi""")


def test_tracker_as_readable_transcript_and_discard_excess_turns_with_default_max_turns(
    domain: Domain,
):
    tracker = DialogueStateTracker(sender_id="test", slots=domain.slots)
    tracker.update_with_events(
        [
            UserUttered("A0"),
            BotUttered("B1"),
            UserUttered("C2"),
            BotUttered("D3"),
            UserUttered("E4"),
            BotUttered("F5"),
            UserUttered("G6"),
            BotUttered("H7"),
            UserUttered("I8"),
            BotUttered("J9"),
            UserUttered("K10"),
            BotUttered("L11"),
            UserUttered("M12"),
            BotUttered("N13"),
            UserUttered("O14"),
            BotUttered("P15"),
            UserUttered("Q16"),
            BotUttered("R17"),
            UserUttered("S18"),
            BotUttered("T19"),
            UserUttered("U20"),
            BotUttered("V21"),
            UserUttered("W22"),
            BotUttered("X23"),
            UserUttered("Y24"),
        ],
        domain,
    )
    response = tracker_as_readable_transcript(tracker)
    assert response == (
        """AI: F5\nUSER: G6\nAI: H7\nUSER: I8\nAI: J9\nUSER: K10\nAI: L11\n"""
        """USER: M12\nAI: N13\nUSER: O14\nAI: P15\nUSER: Q16\nAI: R17\nUSER: S18\n"""
        """AI: T19\nUSER: U20\nAI: V21\nUSER: W22\nAI: X23\nUSER: Y24"""
    )
    assert response.count("\n") == 19


@pytest.mark.parametrize(
    "message, command, expected_response",
    [
        (
            "Very long message",
            {
                "command": "error",
                "error_type": RASA_PATTERN_INTERNAL_ERROR_USER_INPUT_TOO_LONG,
            },
            ERROR_PLACEHOLDER[RASA_PATTERN_INTERNAL_ERROR_USER_INPUT_TOO_LONG],
        ),
        (
            "",
            {
                "command": "error",
                "error_type": RASA_PATTERN_INTERNAL_ERROR_USER_INPUT_EMPTY,
            },
            ERROR_PLACEHOLDER[RASA_PATTERN_INTERNAL_ERROR_USER_INPUT_EMPTY],
        ),
    ],
)
def test_tracker_as_readable_transcript_with_messages_that_triggered_error(
    message: Text,
    command: Dict[Text, Any],
    expected_response: Text,
    domain: Domain,
):
    # Given
    tracker = DialogueStateTracker(sender_id="test", slots=domain.slots)
    tracker.update_with_events(
        [
            UserUttered("Hi"),
            BotUttered("Hi, how can I help you"),
            UserUttered(text=message, parse_data={"commands": [command]}),
            BotUttered("Error response"),
        ]
    )
    # When
    response = tracker_as_readable_transcript(tracker)
    # Then
    assert response == (
        f"USER: Hi\n"
        f"AI: Hi, how can I help you\n"
        f"USER: {expected_response}\n"
        f"AI: Error response"
    )
    assert response.count("\n") == 3


def test_sanitize_message_for_prompt_handles_none():
    assert sanitize_message_for_prompt(None) == ""


def test_sanitize_message_for_prompt_handles_empty_string():
    assert sanitize_message_for_prompt("") == ""


def test_sanitize_message_for_prompt_handles_string_with_newlines():
    assert sanitize_message_for_prompt("hello\nworld") == "hello world"


@pytest.mark.skip(
    reason="This test is going to be updated with the LLM factory update."
)
def test_llm_factory(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test")

    llm = llm_factory(None, {"_type": "openai"})
    assert isinstance(llm, OpenAI)


@pytest.mark.skip(
    reason="This test is going to be updated with the LLM factory update."
)
def test_llm_factory_handles_type_without_underscore(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test")

    llm = llm_factory({"type": "openai"}, {})
    assert isinstance(llm, OpenAI)


@pytest.mark.skip(
    reason="This test is going to be updated with the LLM factory update."
)
def test_llm_factory_uses_custom_type(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test")

    llm = llm_factory({"type": "openai"}, {"_type": "foobar"})
    assert isinstance(llm, OpenAI)


@pytest.mark.skip(
    reason="This test is going to be updated with the LLM factory update."
)
def test_llm_factory_ignores_irrelevant_default_args(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test")

    # since the types of the custom config and the default are different
    # all default arguments should be removed.
    llm = llm_factory({"type": "openai"}, {"_type": "foobar", "temperature": -1})
    assert isinstance(llm, OpenAI)
    # since the default argument should be removed, this should be the default -
    # which is not -1
    assert llm.temperature != -1


@pytest.mark.skip(
    reason="This test is going to be updated with the LLM factory update."
)
def test_llm_factory_fails_on_invalid_args(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test")

    # since the types of the custom config and the default are the same
    # all default arguments should be kept. since the "foo" argument
    # is not a valid argument for the OpenAI class, this should fail
    llm = llm_factory({"type": "openai"}, {"_type": "openai", "temperature": -1})
    assert isinstance(llm, OpenAI)
    # since the default argument should NOT be removed, this should be -1 now
    assert llm.temperature == -1


@pytest.mark.skip(
    reason="This test is going to be updated with the LLM factory update."
)
def test_llm_factory_uses_additional_args_from_custom(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test")

    llm = llm_factory({"temperature": -1}, {"_type": "openai"})
    assert isinstance(llm, OpenAI)
    assert llm.temperature == -1


@pytest.mark.skip(
    reason="This test is going to be updated with the embedding factory update."
)
def test_embedder_factory(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test")

    embedder = embedder_factory(None, {"_type": "openai"})
    assert isinstance(embedder, OpenAIEmbeddings)


@pytest.mark.skip(
    reason="This test is going to be updated with the embedding factory update."
)
def test_embedder_factory_handles_type_without_underscore(
    monkeypatch: MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test")

    embedder = embedder_factory({"type": "openai"}, {})
    assert isinstance(embedder, OpenAIEmbeddings)


@pytest.mark.skip(
    reason="This test is going to be updated with the embedding factory update."
)
def test_embedder_factory_uses_custom_type(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test")

    embedder = embedder_factory({"type": "openai"}, {"_type": "foobar"})
    assert isinstance(embedder, OpenAIEmbeddings)


@pytest.mark.skip(
    reason="This test is going to be updated with the embedding factory update."
)
def test_embedder_factory_ignores_irrelevant_default_args(
    monkeypatch: MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test")

    # embedders don't expect args, they should just be ignored
    embedder = embedder_factory({"type": "openai"}, {"_type": "foobar", "foo": "bar"})
    assert isinstance(embedder, OpenAIEmbeddings)


@pytest.mark.parametrize(
    "input_slot, expected_slot_values",
    [
        (FloatSlot("test_slot", []), None),
        (TextSlot("test_slot", []), None),
        (BooleanSlot("test_slot", []), "[True, False]"),
        (
            CategoricalSlot("test_slot", [], values=["Value1", "Value2"]),
            "['Value1', 'Value2']",
        ),
    ],
)
def test_allowed_values_for_slot(
    input_slot: Slot,
    expected_slot_values: Optional[str],
):
    """Test that allowed_values_for_slot returns the correct values."""
    # When
    allowed_values = allowed_values_for_slot(input_slot)
    # Then
    assert allowed_values == expected_slot_values


@pytest.mark.skip(
    reason="This test is going to be updated with the LLM factory update."
)
def test_llm_factory_azure_openai_models_no_param_transformation(
    monkeypatch: MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test")

    llm = llm_factory(
        {
            "openai_api_type": "azure",
            "openai_api_base": "http://test",
            "openai_api_version": "test_dev",
            "deployment_name": "test",
        },
        {"_type": "openai"},
    )
    assert isinstance(llm, AzureChatOpenAI)
    assert llm.openai_api_type == "azure"
    assert llm.openai_api_base == "http://test/openai"
    assert llm.openai_api_version == "test_dev"
    assert llm.deployment_name == "test"


@pytest.mark.skip(
    reason="This test is going to be updated with the LLM factory update."
)
def test_llm_factory_azure_openai_models_with_api_type(
    monkeypatch: MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test")

    llm = llm_factory(
        {
            "api_type": "azure",
            "openai_api_base": "http://test",
            "openai_api_version": "test_dev",
            "deployment_name": "test",
        },
        {"_type": "openai"},
    )
    assert isinstance(llm, AzureChatOpenAI)
    assert llm.openai_api_type == "azure"
    assert llm.openai_api_base == "http://test"
    assert llm.openai_api_version == "test_dev"
    assert llm.deployment_name == "test"
    assert not hasattr(llm, "api_type")


@pytest.mark.skip(
    reason="This test is going to be updated with the LLM factory update."
)
def test_llm_factory_azure_openai_models_with_api_version(
    monkeypatch: MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test")

    llm = llm_factory(
        {
            "api_version": "test_dev",
            "openai_api_type": "azure",
            "openai_api_base": "http://test",
            "deployment_name": "test",
        },
        {"_type": "openai"},
    )
    assert isinstance(llm, AzureChatOpenAI)
    assert llm.openai_api_type == "azure"
    assert llm.openai_api_base == "http://test"
    assert llm.openai_api_version == "test_dev"
    assert llm.deployment_name == "test"
    assert not hasattr(llm, "api_version")


@pytest.mark.skip(
    reason="This test is going to be updated with the LLM factory update."
)
def test_llm_factory_azure_openai_models_with_api_base(
    monkeypatch: MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test")

    llm = llm_factory(
        {
            "openai_api_type": "azure",
            "api_base": "http://test",
            "openai_api_version": "test_dev",
            "engine": "test",
        },
        {"_type": "openai"},
    )
    assert isinstance(llm, AzureChatOpenAI)
    assert llm.openai_api_type == "azure"
    assert llm.openai_api_base == "http://test"
    assert llm.openai_api_version == "test_dev"
    assert llm.deployment_name == "test"
    assert not hasattr(llm, "api_base")


@pytest.mark.skip(
    reason="This test is going to be updated with the LLM factory update."
)
def test_llm_factory_azure_openai_models_with_deployment(
    monkeypatch: MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test")

    llm = llm_factory(
        {
            "openai_api_type": "azure",
            "openai_api_base": "http://test",
            "openai_api_version": "test_dev",
            "deployment": "test",
        },
        {"_type": "openai"},
    )
    assert isinstance(llm, AzureChatOpenAI)
    assert llm.openai_api_type == "azure"
    assert llm.openai_api_base == "http://test"
    assert llm.openai_api_version == "test_dev"
    assert llm.deployment_name == "test"
    assert not hasattr(llm, "deployment")


@pytest.mark.skip(
    reason="This test is going to be updated with the LLM factory update."
)
def test_llm_factory_azure_openai_models_with_engine(
    monkeypatch: MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test")

    llm = llm_factory(
        {
            "openai_api_type": "azure",
            "openai_api_base": "http://test",
            "openai_api_version": "test_dev",
            "engine": "test",
        },
        {"_type": "openai"},
    )
    assert isinstance(llm, AzureChatOpenAI)
    assert llm.openai_api_type == "azure"
    assert llm.openai_api_base == "http://test"
    assert llm.openai_api_version == "test_dev"
    assert llm.deployment_name == "test"
    assert not hasattr(llm, "engine")


@pytest.mark.skip(
    reason="This test is going to be updated with the LLM factory update."
)
def test_llm_factory_azure_openai_models_with_api_base_in_env(
    monkeypatch: MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    monkeypatch.setenv("OPENAI_API_BASE", "http://test")

    llm = llm_factory(
        {
            "openai_api_type": "azure",
            "openai_api_version": "test_dev",
            "deployment_name": "test",
        },
        {"_type": "openai"},
    )
    assert isinstance(llm, AzureChatOpenAI)
    assert llm.openai_api_type == "azure"
    assert llm.openai_api_version == "test_dev"
    assert llm.deployment_name == "test"
    assert llm.openai_api_base == "http://test"


@pytest.mark.skip(
    reason="This test is going to be updated with the LLM factory update."
)
def test_llm_factory_azure_openai_models_with_api_version_in_env(
    monkeypatch: MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    monkeypatch.setenv("OPENAI_API_VERSION", "test_dev")

    llm = llm_factory(
        {
            "openai_api_type": "azure",
            "openai_api_base": "http://test",
            "deployment_name": "test",
        },
        {"_type": "openai"},
    )
    assert isinstance(llm, AzureChatOpenAI)
    assert llm.openai_api_base == "http://test"
    assert llm.openai_api_type == "azure"
    assert llm.deployment_name == "test"
    assert llm.openai_api_version == "test_dev"


@pytest.mark.skip(
    reason="This test is going to be updated with the LLM factory update."
)
def test_llm_factory_azure_openai_models_with_api_type_in_env(
    monkeypatch: MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    monkeypatch.setenv("OPENAI_API_TYPE", "azure")

    llm = llm_factory(
        {
            "openai_api_base": "http://test",
            "openai_api_version": "test_dev",
            "deployment_name": "test",
        },
        {"_type": "openai"},
    )
    assert isinstance(llm, AzureChatOpenAI)
    assert llm.openai_api_base == "http://test"
    assert llm.deployment_name == "test"
    assert llm.openai_api_version == "test_dev"
    assert llm.openai_api_type == "azure"


@pytest.mark.skip(
    reason="This test is going to be updated with the LLM factory update."
)
def test_llm_factory_azure_openai_models_with_many_env_set(
    monkeypatch: MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    monkeypatch.setenv("OPENAI_API_BASE", "http://test")
    monkeypatch.setenv("OPENAI_API_TYPE", "azure")
    monkeypatch.setenv("OPENAI_API_VERSION", "test_dev")

    llm = llm_factory(
        {
            "deployment_name": "test",
        },
        {"_type": "openai"},
    )
    assert isinstance(llm, AzureChatOpenAI)
    assert llm.deployment_name == "test"
    assert llm.openai_api_base == "http://test"
    assert llm.openai_api_type == "azure"
    assert llm.openai_api_version == "test_dev"


def test_get_prompt_template_returns_default_prompt() -> None:
    default_prompt_template = "default prompt template"
    response = get_prompt_template(None, default_prompt_template)
    assert response == default_prompt_template


def test_get_prompt_template_returns_custom_prompt(tmp_path: Path) -> None:
    prompt_template = "This is a custom prompt template"
    custom_prompt_file = tmp_path / "custom_prompt.jinja2"
    custom_prompt_file.write_text(prompt_template)
    response = get_prompt_template(custom_prompt_file, "default prompt")
    assert response == prompt_template


def test_get_prompt_template_returns_default_on_error() -> None:
    default_prompt_template = "default prompt template"
    response = get_prompt_template("non_existent_file.jinja2", default_prompt_template)
    assert response == default_prompt_template
