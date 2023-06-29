from unittest.mock import patch

import openai.error
from rasa.shared.core.domain import Domain
from rasa.shared.core.events import BotUttered, UserUttered
from rasa.shared.core.trackers import DialogueStateTracker
from rasa.utils.llm import (
    generate_text_openai_chat,
    sanitize_message_for_prompt,
    tracker_as_readable_transcript,
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
        domain,
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


def test_generate_text_openai_chat_handles_exception():
    # mock the openai api in the function openai.ChatCompletion.create
    with patch("openai.ChatCompletion.create") as mock:
        mock.side_effect = openai.error.OpenAIError("test")

        assert generate_text_openai_chat("test", "test") is None


def test_generate_text_openai_chat_handles_response():
    # mock the openai api

    with patch("openai.ChatCompletion.create") as mock:
        # mock a call to `choices[0].message.content`
        mock.return_value = type(
            "obj",
            (object,),
            {
                "choices": [
                    type(
                        "obj",
                        (object,),
                        {"message": type("obj", (object,), {"content": "test"})},
                    )
                ]
            },
        )

        assert generate_text_openai_chat("generate me a test") == "test"
