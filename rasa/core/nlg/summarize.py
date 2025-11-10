from itertools import groupby
from typing import Any, Callable, Dict, List, Optional

import structlog
from jinja2 import Template

from rasa.shared.core.trackers import DialogueStateTracker
from rasa.shared.providers.llm.llm_client import LLMClient
from rasa.shared.utils.constants import (
    LANGFUSE_METADATA_AGENT_ID,
    LANGFUSE_METADATA_COMPONENT_NAME,
    LANGFUSE_METADATA_CUSTOM_METADATA,
    LANGFUSE_METADATA_MODEL_ID,
    LANGFUSE_METADATA_SESSION_ID,
    LANGFUSE_METADATA_TAGS,
)
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


def get_llm_tracing_metadata(tracker: DialogueStateTracker) -> Dict[str, Any]:
    return {
        LANGFUSE_METADATA_SESSION_ID: tracker.sender_id,
        LANGFUSE_METADATA_TAGS: ["ContextualResponseRephraser Summarizer"],
        LANGFUSE_METADATA_CUSTOM_METADATA: {
            LANGFUSE_METADATA_AGENT_ID: tracker.assistant_id,
            LANGFUSE_METADATA_MODEL_ID: tracker.model_id,
            LANGFUSE_METADATA_COMPONENT_NAME: "ContextualResponseRephraser Summarizer",
        },
    }


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
        llm_response = await llm.acompletion(
            prompt, metadata=get_llm_tracing_metadata(tracker)
        )
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
