"""Unit tests for MessageProcessor.handle_voice_conversation."""

import asyncio
from dataclasses import asdict
from typing import Optional
from unittest.mock import AsyncMock, MagicMock

import pytest
import structlog

from rasa.core.channels.constants import (
    USER_CONVERSATION_SESSION_END,
    USER_CONVERSATION_SESSION_START,
    USER_CONVERSATION_SILENCE_TIMEOUT,
)
from rasa.core.channels.conversation_queue.events import (
    BargeInInputEvent,
    DTMFInputEvent,
    FinalTranscriptInputEvent,
    InputEvent,
    SessionEndedInputEvent,
    SessionStartedInputEvent,
    SilenceDetectedInputEvent,
)
from rasa.core.channels.conversation_queue.queue import InMemoryConversationQueue
from rasa.core.channels.voice_ready.utils import CallParameters
from rasa.core.processor import MessageProcessor
from rasa.core.tracker_stores.tracker_store import InMemoryTrackerStore
from rasa.shared.core.domain import Domain
from rasa.shared.core.events import (
    BotUttered,
    UserBargeIn,
    UserUttered,
)
from tests.utilities import filter_logs

_TEST_VOICE_CALL_PARAMETERS = CallParameters(
    call_id="call-1",
    user_phone="+15551000",
    bot_phone="+15552000",
    direction="inbound",
    stream_id="stream-1",
    language="en",
)
_TEST_INPUT_CHANNEL = "test_voice_channel"


@pytest.fixture
def processor() -> MagicMock:
    """Minimal MagicMock wired with the real voice-conversation methods."""
    p = MagicMock(spec=MessageProcessor)
    p.handle_message = AsyncMock()
    p.get_tracker = AsyncMock(return_value=MagicMock())
    p.save_tracker = AsyncMock()
    p.record_event_on_tracker = AsyncMock(
        wraps=MessageProcessor.record_event_on_tracker.__get__(p)
    )
    p.domain = MagicMock()
    p._handle_drained_events = AsyncMock(
        wraps=MessageProcessor._handle_drained_events.__get__(p)
    )
    p.handle_voice_conversation = MessageProcessor.handle_voice_conversation.__get__(p)

    lock_cm = MagicMock()
    lock_cm.__aenter__ = AsyncMock(return_value=None)
    lock_cm.__aexit__ = AsyncMock(return_value=False)
    p.lock_store = MagicMock()
    p.lock_store.lock = MagicMock(return_value=lock_cm)

    return p


def _make_queue(*events: InputEvent) -> InMemoryConversationQueue[InputEvent]:
    """Build a pre-populated InMemoryConversationQueue."""
    q: InMemoryConversationQueue[InputEvent] = InMemoryConversationQueue(
        conversation_id="test-conv",
        input_channel=_TEST_INPUT_CHANNEL,
        maxsize=50,
    )
    for ev in events:
        q._queue.put_nowait(ev)
    return q


class TestHandleVoiceConversation:
    """Tests for MessageProcessor.handle_voice_conversation loop control."""

    @pytest.mark.asyncio
    async def test_exits_on_session_ended(self, processor: MagicMock) -> None:
        """Loop returns after receiving a SessionEndedInputEvent."""
        q = _make_queue(SessionEndedInputEvent())
        await processor.handle_voice_conversation(q, MagicMock(), "sender-1")

    @pytest.mark.asyncio
    async def test_processes_multiple_turns_before_session_ended(
        self, processor: MagicMock
    ) -> None:
        """Loop runs as many turns as needed before SessionEndedInputEvent."""
        q = _make_queue(
            FinalTranscriptInputEvent("turn one"),
            FinalTranscriptInputEvent("turn two"),
            SessionEndedInputEvent(),
        )

        await processor.handle_voice_conversation(q, MagicMock(), "sender-1")

        # All three events were in one drain batch: transcripts get coalesced into a
        # single `UserMessage`, and `SessionEndedInputEvent` is routed through
        # `handle_message` as `/session_end`.
        assert processor.handle_message.await_count == 2
        assert processor.handle_message.call_args_list[0].args[0].text == (
            "turn one turn two"
        )
        assert processor.handle_message.call_args_list[1].args[0].text == (
            USER_CONVERSATION_SESSION_END
        )

    @pytest.mark.asyncio
    async def test_events_arriving_in_separate_turns_produce_separate_messages(
        self, processor: MagicMock
    ) -> None:
        """Events arriving in separate turns each produce a handle_message call."""
        q: InMemoryConversationQueue[InputEvent] = InMemoryConversationQueue(
            conversation_id="test-conv",
            input_channel=_TEST_INPUT_CHANNEL,
            maxsize=10,
        )

        async def _producer() -> None:
            await q.put(FinalTranscriptInputEvent("turn one"))
            # Yield so the loop drains and processes turn one before turn two arrives.
            await asyncio.sleep(0.05)
            await q.put(FinalTranscriptInputEvent("turn two"))
            await asyncio.sleep(0.05)
            await q.put(FinalTranscriptInputEvent("turn three"))
            await asyncio.sleep(0.05)
            await q.put(SessionEndedInputEvent())

        await asyncio.gather(
            processor.handle_voice_conversation(q, MagicMock(), "sender-1"),
            _producer(),
        )

        # Three transcript turns + one `/session_end` lifecycle message.
        assert processor.handle_message.await_count == 4
        assert processor.handle_message.call_args_list[0].args[0].text == "turn one"
        assert processor.handle_message.call_args_list[1].args[0].text == "turn two"
        assert processor.handle_message.call_args_list[2].args[0].text == "turn three"
        assert processor.handle_message.call_args_list[3].args[0].text == (
            USER_CONVERSATION_SESSION_END
        )


class TestHandleDrainedEvents:
    """Tests for MessageProcessor._handle_drained_events event routing."""

    @pytest.mark.asyncio
    async def test_single_transcript_produces_one_user_message(
        self, processor: MagicMock
    ) -> None:
        """A single FinalTranscriptInputEvent results in one handle_message call."""
        await MessageProcessor._handle_drained_events(
            processor,
            [FinalTranscriptInputEvent("hello")],
            MagicMock(),
            "s1",
            _TEST_INPUT_CHANNEL,
        )

        processor.handle_message.assert_awaited_once()
        msg = processor.handle_message.call_args.args[0]
        assert msg.text == "hello"
        assert msg.sender_id == "s1"
        assert msg.input_channel == _TEST_INPUT_CHANNEL

    @pytest.mark.asyncio
    async def test_multiple_transcripts_are_space_joined(
        self, processor: MagicMock
    ) -> None:
        """FinalTranscriptInputEvent texts are space-joined into one message."""
        events = [
            FinalTranscriptInputEvent("hello"),
            FinalTranscriptInputEvent("there"),
            FinalTranscriptInputEvent("world"),
        ]
        await MessageProcessor._handle_drained_events(
            processor, events, MagicMock(), "s1", _TEST_INPUT_CHANNEL
        )

        processor.handle_message.assert_awaited_once()
        msg = processor.handle_message.call_args.args[0]
        assert msg.text == "hello there world"

    @pytest.mark.asyncio
    async def test_empty_batch_skips_handle_message(self, processor: MagicMock) -> None:
        """A batch with no FinalTranscriptInputEvent events skips handle_message."""
        await MessageProcessor._handle_drained_events(
            processor, [], MagicMock(), "s1", _TEST_INPUT_CHANNEL
        )

        processor.handle_message.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_barge_in_input_event_is_recorded_on_tracker(
        self, processor: MagicMock
    ) -> None:
        """BargeInInputEvent is recorded as the only direct tracker write.

        It has no NLU placeholder pattern, so the processor appends the
        `UserBargeIn` event itself rather than routing through `handle_message`.
        """
        await MessageProcessor._handle_drained_events(
            processor, [BargeInInputEvent()], MagicMock(), "s1", _TEST_INPUT_CHANNEL
        )

        tracker = processor.get_tracker.return_value
        tracker.update.assert_called_once()
        assert isinstance(tracker.update.call_args.args[0], UserBargeIn)
        processor.save_tracker.assert_awaited_once_with(tracker)
        processor.handle_message.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_barge_in_input_event_emits_info_log(
        self, processor: MagicMock
    ) -> None:
        """BargeInInputEvent emits a structlog info event."""
        with structlog.testing.capture_logs() as caplog:
            await MessageProcessor._handle_drained_events(
                processor,
                [BargeInInputEvent()],
                MagicMock(),
                "s1",
                _TEST_INPUT_CHANNEL,
            )

        info_logs = filter_logs(
            caplog,
            event="processor.handle_voice_conversation.user_barge_in",
            log_level="info",
        )
        assert len(info_logs) == 1
        assert info_logs[0]["sender_id"] == "s1"

    @pytest.mark.asyncio
    async def test_silence_detected_input_event_emits_info_log(
        self, processor: MagicMock
    ) -> None:
        """SilenceDetectedInputEvent emits a structlog info event."""
        with structlog.testing.capture_logs() as caplog:
            await MessageProcessor._handle_drained_events(
                processor,
                [SilenceDetectedInputEvent()],
                MagicMock(),
                "s1",
                _TEST_INPUT_CHANNEL,
            )

        info_logs = filter_logs(
            caplog,
            event="processor.handle_voice_conversation.silence_detected",
            log_level="info",
        )
        assert len(info_logs) == 1
        assert info_logs[0]["sender_id"] == "s1"

    @pytest.mark.parametrize(
        "input_event,expected_text,expected_metadata",
        [
            (
                SessionStartedInputEvent(metadata=asdict(_TEST_VOICE_CALL_PARAMETERS)),
                USER_CONVERSATION_SESSION_START,
                asdict(_TEST_VOICE_CALL_PARAMETERS),
            ),
            (
                SessionEndedInputEvent(),
                USER_CONVERSATION_SESSION_END,
                None,
            ),
            (
                SilenceDetectedInputEvent(),
                USER_CONVERSATION_SILENCE_TIMEOUT,
                None,
            ),
            (
                DTMFInputEvent("123"),
                "123",
                {"source": "dtmf"},
            ),
        ],
        ids=["session_started", "session_ended", "silence_detected", "dtmf_input"],
    )
    @pytest.mark.asyncio
    async def test_lifecycle_input_event_routes_through_handle_message(
        self,
        processor: MagicMock,
        input_event: InputEvent,
        expected_text: str,
        expected_metadata: dict,
    ) -> None:
        """Lifecycle / DTMF / silence input events become `UserMessage`s.

        These events have placeholder-intent counterparts (`/session_start`,
        `/session_end`, `/silence_timeout`, or the raw digits) and run through
        the normal NLU + pattern pipeline rather than being appended directly
        to the tracker.
        """
        await MessageProcessor._handle_drained_events(
            processor, [input_event], MagicMock(), "s1", _TEST_INPUT_CHANNEL
        )

        processor.handle_message.assert_awaited_once()
        message = processor.handle_message.call_args.args[0]
        assert message.text == expected_text
        assert message.metadata == expected_metadata
        assert message.sender_id == "s1"
        assert message.input_channel == _TEST_INPUT_CHANNEL

        tracker = processor.get_tracker.return_value
        tracker.update.assert_not_called()

    @pytest.mark.asyncio
    async def test_pending_transcripts_flush_before_session_ended(
        self, processor: MagicMock
    ) -> None:
        """Accumulated transcripts run before a trailing SessionEndedInputEvent.

        Otherwise `/session_end` would terminate the tracker before the
        transcript bot turn can be dispatched against it.
        """
        await MessageProcessor._handle_drained_events(
            processor,
            [
                BargeInInputEvent(),
                FinalTranscriptInputEvent("hello"),
                FinalTranscriptInputEvent("there"),
                SessionEndedInputEvent(),
            ],
            MagicMock(),
            "s1",
            _TEST_INPUT_CHANNEL,
        )

        tracker = processor.get_tracker.return_value
        recorded_events = [call.args[0] for call in tracker.update.call_args_list]
        assert [type(event) for event in recorded_events] == [UserBargeIn]
        assert processor.save_tracker.await_count == 1

        assert processor.handle_message.await_count == 2
        first_message, second_message = (
            processor.handle_message.call_args_list[0].args[0],
            processor.handle_message.call_args_list[1].args[0],
        )
        assert first_message.text == "hello there"
        assert second_message.text == USER_CONVERSATION_SESSION_END


class TestLockAcquisition:
    """Tests verifying per-turn (not per-call) lock acquisition."""

    @pytest.mark.asyncio
    async def test_lock_acquired_once_for_single_turn(
        self, processor: MagicMock
    ) -> None:
        """lock_store.lock is called exactly once for a single-turn conversation."""
        q = _make_queue(FinalTranscriptInputEvent("hi"), SessionEndedInputEvent())

        await processor.handle_voice_conversation(q, MagicMock(), "conv-id")

        processor.lock_store.lock.assert_called_once_with("conv-id")

    @pytest.mark.asyncio
    async def test_lock_acquired_once_per_drain_cycle(
        self, processor: MagicMock
    ) -> None:
        """lock_store.lock is called once per drain cycle across multiple turns."""
        q: InMemoryConversationQueue[InputEvent] = InMemoryConversationQueue(
            conversation_id="c",
            input_channel=_TEST_INPUT_CHANNEL,
            maxsize=10,
        )
        await q.put(FinalTranscriptInputEvent("turn one"))
        await q.put(SessionEndedInputEvent())

        await processor.handle_voice_conversation(q, MagicMock(), "conv-id")

        # Both events were drained in a single batch (get + drain in one iteration),
        # so lock is called exactly once.
        processor.lock_store.lock.assert_called_once_with("conv-id")

    @pytest.mark.asyncio
    async def test_lock_acquired_once_per_separate_turn(
        self, processor: MagicMock
    ) -> None:
        """lock_store.lock is called once per turn when turns arrive separately."""
        q: InMemoryConversationQueue[InputEvent] = InMemoryConversationQueue(
            conversation_id="c",
            input_channel=_TEST_INPUT_CHANNEL,
            maxsize=10,
        )

        async def _producer() -> None:
            await q.put(FinalTranscriptInputEvent("turn one"))
            await asyncio.sleep(0.05)
            await q.put(FinalTranscriptInputEvent("turn two"))
            await q.put(FinalTranscriptInputEvent("turn two two"))
            await asyncio.sleep(0.05)
            await q.put(FinalTranscriptInputEvent("turn three"))
            await q.put(FinalTranscriptInputEvent("turn three three three"))
            await asyncio.sleep(0.05)
            await q.put(SessionEndedInputEvent())

        await asyncio.gather(
            processor.handle_voice_conversation(q, MagicMock(), "conv-id"),
            _producer(),
        )

        # Four separate drain cycles: turn one, turn two (batched), turn three
        # (batched), session ended — lock acquired once per cycle.
        assert processor.lock_store.lock.call_count == 4
        processor.lock_store.lock.assert_called_with("conv-id")

        # Three transcript turns plus one `/session_end` lifecycle message.
        assert processor.handle_message.await_count == 4
        texts = [c.args[0].text for c in processor.handle_message.call_args_list]
        assert texts == [
            "turn one",
            "turn two turn two two",
            "turn three turn three three three",
            USER_CONVERSATION_SESSION_END,
        ]


class TestTrackerDurabilityAcrossHandleMessage:
    """End-to-end style checks for tracker event durability inside a drain batch.

    `handle_message` loads its own tracker from the store and saves it back, so
    any reference held by `_handle_drained_events` goes stale across iterations.
    These tests wire a real `InMemoryTrackerStore` so a regression — e.g.
    removing the per-iteration re-fetch — would surface as lost or reordered
    events in the persisted tracker.
    """

    @pytest.fixture
    def processor_with_real_store(self) -> MagicMock:
        """Processor wired to a real `InMemoryTrackerStore`.

        `handle_message` is faked to mirror its real interaction with the store
        (load fresh tracker, append events, save), which is precisely the
        behavior that makes any local tracker reference go stale.
        """
        domain = Domain.empty()
        tracker_store = InMemoryTrackerStore(domain)

        processor = MagicMock(spec=MessageProcessor)
        processor.domain = domain
        processor.tracker_store = tracker_store

        async def _get_tracker(sender_id: str, user_id: Optional[str] = None) -> object:
            return await tracker_store.get_or_create_tracker(
                sender_id, append_action_listen=False, user_id=user_id
            )

        async def _save_tracker(tracker: object) -> None:
            await tracker_store.save(tracker)  # type: ignore[arg-type]

        async def _fake_handle_message(message: object) -> None:
            tracker = await tracker_store.get_or_create_tracker(
                message.sender_id,  # type: ignore[attr-defined]
                append_action_listen=False,
            )
            tracker.update(
                UserUttered(message.text),  # type: ignore[attr-defined]
                domain,
            )
            tracker.update(
                BotUttered(f"response to {message.text}"),  # type: ignore[attr-defined]
                domain,
            )
            await tracker_store.save(tracker)

        processor.get_tracker = AsyncMock(side_effect=_get_tracker)
        processor.save_tracker = AsyncMock(side_effect=_save_tracker)
        processor.handle_message = AsyncMock(side_effect=_fake_handle_message)
        processor.record_event_on_tracker = AsyncMock(
            wraps=MessageProcessor.record_event_on_tracker.__get__(processor)
        )

        return processor

    @pytest.mark.parametrize(
        "input_events, expected_event_types, expected_user_text",
        [
            (
                [BargeInInputEvent()],
                ["UserBargeIn"],
                None,
            ),
            (
                [FinalTranscriptInputEvent("hi"), BargeInInputEvent()],
                ["UserUttered", "BotUttered", "UserBargeIn"],
                "hi",
            ),
            (
                [
                    FinalTranscriptInputEvent("hi"),
                    FinalTranscriptInputEvent("there"),
                    BargeInInputEvent(),
                ],
                ["UserUttered", "BotUttered", "UserBargeIn"],
                "hi there",
            ),
            (
                [BargeInInputEvent(), FinalTranscriptInputEvent("hello")],
                ["UserBargeIn", "UserUttered", "BotUttered"],
                "hello",
            ),
            (
                [
                    BargeInInputEvent(),
                    FinalTranscriptInputEvent("hi"),
                    FinalTranscriptInputEvent("there"),
                    BargeInInputEvent(),
                ],
                [
                    "UserBargeIn",
                    "UserUttered",
                    "BotUttered",
                    "UserBargeIn",
                ],
                "hi there",
            ),
        ],
        ids=[
            "barge_in_only",
            "single_transcript_then_barge_in",
            "coalesced_transcripts_then_barge_in",
            "barge_in_then_transcript",
            "coalesced_transcripts_between_barge_ins",
        ],
    )
    @pytest.mark.asyncio
    async def test_drained_batch_persists_events_in_order(
        self,
        processor_with_real_store: MagicMock,
        input_events: list,
        expected_event_types: list,
        expected_user_text: Optional[str],
    ) -> None:
        """Drained batches persist all events in order, including coalesced transcripts.

        Regressions guarded:

        * Holding a stale local tracker across iterations would cause a trailing
          `record_event_on_tracker` save to overwrite events `handle_message` just
          persisted, dropping `UserUttered` and `BotUttered` from the stored event log.
        * Removing the save inside `record_event_on_tracker` would silently drop
          drop tracker-only batches (e.g. `BargeIn` alone) when the lock releases
          without `handle_message` ever running.
        * Consecutive `FinalTranscriptInputEvent`s are coalesced into a single
          `UserUttered` whose text is the space-joined concatenation.

        `expected_user_text` is ``None`` when the batch contains no transcript.
        """
        await MessageProcessor._handle_drained_events(
            processor_with_real_store,
            input_events,
            MagicMock(),
            "s1",
            _TEST_INPUT_CHANNEL,
        )

        persisted = await processor_with_real_store.tracker_store.retrieve("s1")
        assert persisted is not None
        assert [
            type(event).__name__ for event in persisted.events
        ] == expected_event_types

        user_uttered_events = [
            e for e in persisted.events if isinstance(e, UserUttered)
        ]
        if expected_user_text is None:
            assert user_uttered_events == []
        else:
            assert len(user_uttered_events) == 1
            assert user_uttered_events[0].text == expected_user_text
