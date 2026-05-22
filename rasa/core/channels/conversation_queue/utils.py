from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from rasa.core.channels.conversation_queue.events import (
    FinalTranscriptInputEvent,
    InputEvent,
)


def coalesce_final_transcripts(events: Sequence[InputEvent]) -> List[InputEvent]:
    """Collapse consecutive `FinalTranscriptInputEvent`s into one per run.

    Each maximal run of `FinalTranscriptInputEvent`s in `events` is replaced by a single
    `FinalTranscriptInputEvent` whose `text` is the space-joined concatenation of the
    run and whose `metadata` is the metadata of the last transcript in the run.

    All non-transcript events keep their identity and relative position.

    Args:
        events: The events to coalesce.

    Returns:
        The coalesced events.
    """
    result: List[InputEvent] = []
    pending_texts: List[str] = []
    pending_metadata: Optional[Dict[str, Any]] = None

    def commit_pending_transcripts() -> None:
        """Concatenate buffered transcript texts into one event. Append it to result."""
        nonlocal pending_texts, pending_metadata

        if not pending_texts:
            return

        # Concatenate the buffered transcript texts into one event.
        result.append(
            FinalTranscriptInputEvent(
                text=" ".join(pending_texts), metadata=pending_metadata
            )
        )

        # Clear the buffered transcript texts and metadata for the next batch.
        pending_texts.clear()
        pending_metadata = None

    for event in events:
        if isinstance(event, FinalTranscriptInputEvent):
            pending_texts.append(event.text)
            pending_metadata = event.metadata
        else:
            commit_pending_transcripts()
            result.append(event)

    # Commit any trailing transcripts not followed by a non-transcript event.
    commit_pending_transcripts()

    return result
