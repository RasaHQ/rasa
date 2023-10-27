from rasa.shared.core.domain import Domain
from rasa.shared.core.events import BotUttered, UserUttered
from rasa.shared.core.trackers import DialogueStateTracker
from rasa.shared.utils.llm import (
    sanitize_message_for_prompt,
    tracker_as_readable_transcript,
    embedder_factory,
    llm_factory,
)
from langchain import OpenAI
from langchain.embeddings import OpenAIEmbeddings
from pytest import MonkeyPatch


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


def test_sanitize_message_for_prompt_handles_none():
    assert sanitize_message_for_prompt(None) == ""


def test_sanitize_message_for_prompt_handles_empty_string():
    assert sanitize_message_for_prompt("") == ""


def test_sanitize_message_for_prompt_handles_string_with_newlines():
    assert sanitize_message_for_prompt("hello\nworld") == "hello world"


def test_llm_factory(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test")

    llm = llm_factory(None, {"_type": "openai"})
    assert isinstance(llm, OpenAI)


def test_llm_factory_handles_type_without_underscore(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test")

    llm = llm_factory({"type": "openai"}, {})
    assert isinstance(llm, OpenAI)


def test_llm_factory_uses_custom_type(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test")

    llm = llm_factory({"type": "openai"}, {"_type": "foobar"})
    assert isinstance(llm, OpenAI)


def test_llm_factory_ignores_irrelevant_default_args(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test")

    # since the types of the custom config and the default are different
    # all default arguments should be removed.
    llm = llm_factory({"type": "openai"}, {"_type": "foobar", "temperature": -1})
    assert isinstance(llm, OpenAI)
    # since the default argument should be removed, this should be the default -
    # which is not -1
    assert llm.temperature != -1


def test_llm_factory_fails_on_invalid_args(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test")

    # since the types of the custom config and the default are the same
    # all default arguments should be kept. since the "foo" argument
    # is not a valid argument for the OpenAI class, this should fail
    llm = llm_factory({"type": "openai"}, {"_type": "openai", "temperature": -1})
    assert isinstance(llm, OpenAI)
    # since the default argument should NOT be removed, this should be -1 now
    assert llm.temperature == -1


def test_llm_factory_uses_additional_args_from_custom(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test")

    llm = llm_factory({"temperature": -1}, {"_type": "openai"})
    assert isinstance(llm, OpenAI)
    assert llm.temperature == -1


def test_embedder_factory(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test")

    embedder = embedder_factory(None, {"_type": "openai"})
    assert isinstance(embedder, OpenAIEmbeddings)


def test_embedder_factory_handles_type_without_underscore(
    monkeypatch: MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test")

    embedder = embedder_factory({"type": "openai"}, {})
    assert isinstance(embedder, OpenAIEmbeddings)


def test_embedder_factory_uses_custom_type(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test")

    embedder = embedder_factory({"type": "openai"}, {"_type": "foobar"})
    assert isinstance(embedder, OpenAIEmbeddings)


def test_embedder_factory_ignores_irrelevant_default_args(
    monkeypatch: MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test")

    # embedders don't expect args, they should just be ignored
    embedder = embedder_factory({"type": "openai"}, {"_type": "foobar", "foo": "bar"})
    assert isinstance(embedder, OpenAIEmbeddings)
