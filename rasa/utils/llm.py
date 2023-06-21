from typing import Optional
import openai
import openai.error
import structlog
from rasa.shared.core.events import BotUttered, UserUttered

from rasa.shared.core.trackers import DialogueStateTracker

structlogger = structlog.get_logger()

USER = "USER"

AI = "AI"

DEFAULT_OPENAI_GENERATE_MODEL_NAME = "text-davinci-003"

DEFAULT_OPENAI_CHAT_MODEL_NAME = "gpt-3.5-turbo"

DEFAULT_OPENAI_TEMPERATURE = 0.7


def generate_text_openai_chat(
    prompt: str,
    model: str = DEFAULT_OPENAI_CHAT_MODEL_NAME,
    temperature: float = DEFAULT_OPENAI_TEMPERATURE,
) -> Optional[str]:
    """Generates text using the OpenAI chat API.

    Args:
        prompt: the prompt to send to the API
        model: the model to use for generation
        temperature: the temperature to use for generation

    Returns:
        The generated text.
    """
    # TODO: exception handling
    try:
        chat_completion = openai.ChatCompletion.create(  # type: ignore[no-untyped-call]
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
        )
        return chat_completion.choices[0].message.content
    except openai.error.OpenAIError:
        structlogger.exception("openai.generate.error", model=model, prompt=prompt)
        return None


def tracker_as_readable_transcript(
    tracker: DialogueStateTracker, human_prefix: str = USER, ai_prefix: str = AI
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
        if isinstance(event, BotUttered):
            transcript.append(f"{ai_prefix}: {sanitize_message_for_prompt(event.text)}")
    return "\n".join(transcript)


def sanitize_message_for_prompt(text: Optional[str]) -> str:
    """Removes new lines from a string.

    Args:
        text: the text to sanitize

    Returns:
    A string with new lines removed.
    """
    return text.replace("\n", " ") if text else ""
