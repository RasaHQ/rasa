import asyncio
import uuid

import pytest

from rasa.builder.job_manager import JobInfo, JobManager, job_manager
from rasa.builder.models import JobStatus, JobStatusEvent, ServerSentEventType


async def collect_events(stream):
    """Helper to exhaust an async generator and return a list of items."""
    return [event async for event in stream]


@pytest.mark.asyncio
async def test_create_and_get_job():
    jm = JobManager()
    job = jm.create_job()

    # Raises error if job.id is not a valid UUID
    uuid.UUID(job.id)

    same_job = jm.get_job(job.id)
    assert same_job is job


@pytest.mark.asyncio
async def test_mark_done():
    job = job_manager.create_job()
    job_manager.mark_done(job)

    event = await job._queue.get()
    assert event == JobStatusEvent.eof()


@pytest.mark.asyncio
async def test_mark_done_pushes_eof_and_sets_error():
    job = job_manager.create_job()

    assert job._queue.empty()

    error_message = "error"
    job_manager.mark_done(job, error=error_message)

    assert job.error == error_message

    # The first item in the queue must be _EOF
    eof = await job._queue.get()
    assert isinstance(eof, JobStatusEvent)
    assert eof.event == ServerSentEventType._EOF.value
    assert eof.data == {}


@pytest.mark.asyncio
async def test_sse_build_and_eof_helpers():
    event_ok = JobStatusEvent.from_status(JobStatus.training)
    event_error = JobStatusEvent.from_status(JobStatus.train_error, "error")
    event_eof = JobStatusEvent.eof()

    assert event_ok.event == ServerSentEventType.progress.value
    assert event_ok.data == {"status": JobStatus.training}

    assert event_error.event == ServerSentEventType.error.value
    assert event_error.data == {"status": JobStatus.train_error, "message": "error"}

    assert event_eof.event == ServerSentEventType._EOF.value
    assert event_eof.data == {}


@pytest.mark.asyncio
async def test_event_stream_replays_and_stops_on_eof():
    job = JobInfo(id="test")

    # put two events and an EOF into the queue *before* we start streaming
    await job.put(JobStatusEvent.from_status(JobStatus.generating))
    await job.put(JobStatusEvent.from_status(JobStatus.training))
    await job.put(JobStatusEvent.eof())

    stream_task = asyncio.create_task(collect_events(job.event_stream()))
    events = await asyncio.wait_for(stream_task, timeout=1.0)

    # we must get the two progress events in order, but not the EOF
    assert [e.data["status"] for e in events] == [
        JobStatus.generating,
        JobStatus.training,
    ]

    # history should now contain three items (two progress, one EOF)
    assert len(job._history) == 3
    assert job._history[-1].event == "_EOF"


@pytest.mark.asyncio
async def test_event_stream_stops_if_eof_already_in_history():
    job = JobInfo(id="test-history-stop")
    done_event = JobStatusEvent.from_status(JobStatus.done)
    job._history = [done_event, JobStatusEvent.eof()]

    # Everything until EOF should be streamed
    events = [e async for e in job.event_stream()]
    assert events == [done_event]


@pytest.mark.asyncio
async def test_live_events_are_forwarded_and_added_to_history():
    job = JobInfo(id="live")

    consumer_task = asyncio.create_task(collect_events(job.event_stream()))
    await job.put(JobStatusEvent.from_status(JobStatus.training))
    await job.put(JobStatusEvent.from_status(JobStatus.train_success))
    await job.put(JobStatusEvent.eof())

    events = await asyncio.wait_for(consumer_task, timeout=1.0)
    statuses = [e.data["status"] for e in events]
    assert statuses == [JobStatus.training, JobStatus.train_success]

    assert len(job._history) == 3
    assert job._history[-1].event == ServerSentEventType._EOF.value


@pytest.mark.asyncio
async def test_create_job_with_commit_sha():
    jm = JobManager()
    commit_sha = "abc123def456"
    job = jm.create_job(commit_sha=commit_sha)

    assert job.commit_sha == commit_sha

    # Verify it's still retrievable
    same_job = jm.get_job(job.id)
    assert same_job.commit_sha == commit_sha


@pytest.mark.asyncio
async def test_create_job_without_commit_sha():
    jm = JobManager()
    job = jm.create_job()

    # Should default to None
    assert job.commit_sha is None
