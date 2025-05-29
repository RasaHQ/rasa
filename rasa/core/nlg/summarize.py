from itertools import groupby
from typing import Callable, List, Optional

import structlog
from jinja2 import Template

from rasa.shared.core.trackers import DialogueStateTracker
from rasa.shared.providers.llm.llm_client import LLMClient
from rasa.shared.utils.llm import (
    tracker_as_readable_transcript,
)

structlogger = structlog.get_logger()

_DEFAULT_SUMMARIZER_TEMPLATE = """Summarize the provided conversation between
a user and a conversational AI. The summary should be a short text that
captures the main points of the conversation.

Conversation:
{{conversation}}

Summary:"""
SUMMARY_PROMPT_TEMPLATE = Template(_DEFAULT_SUMMARIZER_TEMPLATE)
MAX_TURNS_DEFAULT = 20


def _count_multiple_utterances_as_single_turn(transcript: List[str]) -> List[str]:
    """Counts multiple utterances as a single turn.

    Args:
        transcript: the lines of the transcript

    Returns:
        transcript: with multiple utterances counted as a single turn
    """
    if not transcript:
        return []

    def get_speaker_label(line: str) -> str:
        return line.partition(": ")[0] if ": " in line else ""

    modified_transcript = [
        f"{speaker}: {' '.join(line.partition(': ')[2] for line in group)}"
        for speaker, group in groupby(transcript, key=get_speaker_label)
        if speaker
    ]

    return modified_transcript


def _create_summarization_prompt(
    tracker: DialogueStateTracker,
    max_turns: Optional[int],
    turns_wrapper: Optional[Callable[[List[str]], List[str]]],
) -> str:
    """Creates an LLM prompt to summarize the conversation in the tracker.

    Args:
        tracker: tracker of the conversation to be summarized
        max_turns: maximum number of turns to summarize
        turns_wrapper: optional function to wrap the turns


    Returns:
        The prompt to summarize the conversation.
    """
    transcript = tracker_as_readable_transcript(
        tracker, max_turns=max_turns, turns_wrapper=turns_wrapper
    )
    return SUMMARY_PROMPT_TEMPLATE.render(
        conversation=transcript,
    )


async def summarize_conversation(
    tracker: DialogueStateTracker,
    llm: LLMClient,
    max_turns: Optional[int] = MAX_TURNS_DEFAULT,
    turns_wrapper: Optional[Callable[[List[str]], List[str]]] = None,
) -> str:
    """Summarizes the dialogue using the LLM.

    Args:
        tracker: the tracker to summarize
        llm: the LLM to use for summarization
        max_turns: maximum number of turns to summarize
        turns_wrapper: optional function to wrap the turns

    Returns:
        The summary of the dialogue.
    """
    prompt = _create_summarization_prompt(tracker, max_turns, turns_wrapper)
    try:
        llm_response = await llm.acompletion(prompt)
        summarization = llm_response.choices[0].strip()
        structlogger.debug(
            "summarization.success", summarization=summarization, prompt=prompt
        )
        return summarization
    except Exception as e:
        transcript = tracker_as_readable_transcript(
            tracker, max_turns=max_turns, turns_wrapper=turns_wrapper
        )
        structlogger.error("summarization.error", error=e)
        return transcript
