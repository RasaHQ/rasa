from typing import Optional
import structlog
from rasa.shared.core.events import BotUttered, UserUttered

from rasa.shared.core.trackers import DialogueStateTracker

structlogger = structlog.get_logger()

USER = "USER"

AI = "AI"

DEFAULT_OPENAI_GENERATE_MODEL_NAME = "text-davinci-003"

DEFAULT_OPENAI_CHAT_MODEL_NAME = "gpt-3.5-turbo"

DEFAULT_OPENAI_EMBEDDING_MODEL_NAME = "text-embedding-ada-002"

DEFAULT_OPENAI_TEMPERATURE = 0.7


def tracker_as_readable_transcript(
    tracker: DialogueStateTracker,
    human_prefix: str = USER,
    ai_prefix: str = AI,
    max_turns: Optional[int] = 20,
) -> str:
    """Creates a readable dialogue from a tracker.

    Args:
        tracker: the tracker to convert
        human_prefix: the prefix to use for human utterances
        ai_prefix: the prefix to use for ai utterances
        max_turns: the maximum number of turns to include in the transcript

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
        USER: hello
        AI: hi

    Returns:
    A string representing the transcript of the tracker
    """
    transcript = []

    for event in tracker.events:
        if isinstance(event, UserUttered):
            transcript.append(
                f"{human_prefix}: {sanitize_message_for_prompt(event.text)}"
            )
        elif isinstance(event, BotUttered):
            transcript.append(f"{ai_prefix}: {sanitize_message_for_prompt(event.text)}")

    if max_turns:
        transcript = transcript[-max_turns:]
    return "\n".join(transcript)


def sanitize_message_for_prompt(text: Optional[str]) -> str:
    """Removes new lines from a string.

    Args:
        text: the text to sanitize

    Returns:
    A string with new lines removed.
    """
    return text.replace("\n", " ") if text else ""
