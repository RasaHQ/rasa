from typing import Optional
import openai
import logging

from rasa_sdk import Tracker

from rasa.shared.core.trackers import DialogueStateTracker

logger = logging.getLogger(__name__)


def generate_text_openai_chat(
    prompt: str, model: str = "gpt-3.5-turbo"
) -> Optional[str]:
    chat_completion = openai.ChatCompletion.create(
        model=model, messages=[{"role": "user", "content": prompt}], temperature=0.0
    )
    return chat_completion.choices[0].message.content


def tracker_as_readable_transcript(
    tracker: DialogueStateTracker, human_prefix: str = "USER", ai_prefix: str = "AI"
) -> str:
    """Creates a readable dialogue from a tracker.

    Args:
        tracker: the tracker to convert
        human_prefix: the prefix to use for human utterances
        ai_prefix: the prefix to use for ai utterances

    Example:
        >>> tracker = Tracker(
        ...     sender_id="test",
        ...     slots=[],
        ...     events=[
        ...         UserUttered("hello"),
        ...         BotUttered("hi"),
        ...     ],
        ... )
        >>> tracker_as_readable_transcript(tracker)
        HUMAN: hello
        AI: hi

    Returns:
        A string representing the transcript of the tracker"""
    transcript = []

    for event in tracker.events:
        if event.type_name == "user":
            transcript.append(
                f"{human_prefix}: {sanitize_message_for_prompt(event.text)}"
            )
        if event.type_name == "bot":
            transcript.append(f"{ai_prefix}: {sanitize_message_for_prompt(event.text)}")
    return "\n".join(transcript)


def sanitize_message_for_prompt(text: Optional[str]) -> str:
    """Removes new lines from a string.

    Args:
        text: the text to sanitize

    Returns:
        A string with new lines removed."""
    return text.replace("\n", " ") if text else ""
