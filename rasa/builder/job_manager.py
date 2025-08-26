import asyncio
import time
import uuid
from typing import AsyncGenerator, ClassVar, Dict, List, Optional

from pydantic import BaseModel, Field, PrivateAttr

from rasa.builder.models import JobStatusEvent, ServerSentEventType


class JobInfo(BaseModel):
    """Information about a job being processed by the worker."""

    id: str
    status: str = ""
    error: Optional[str] = None
    created_at: float = Field(default_factory=time.time)

    _history: List[JobStatusEvent] = PrivateAttr(default_factory=list)
    _queue: asyncio.Queue = PrivateAttr(default_factory=asyncio.Queue)

    class Config:
        arbitrary_types_allowed = True

    async def put(self, event: JobStatusEvent) -> None:
        """Put an event onto the job's queue.

        Args:
            event: The JobStatusEvent to put onto the queue.
        """
        await self._queue.put(event)

    async def event_stream(self) -> AsyncGenerator[JobStatusEvent, None]:
        """Yield events as they are put on the queue by the worker task.

        Returns:
            An async generator that yields JobStatusEvent objects.
        """
        # 1) Replay history and stop if EOF recorded
        for sse in self._history:
            if (
                isinstance(sse, JobStatusEvent)
                and sse.event == ServerSentEventType._EOF.value
            ):
                return
            yield sse

        # 2) Stream live events and stop on EOF
        while True:
            sse = await self._queue.get()
            self._history.append(sse)
            if (
                isinstance(sse, JobStatusEvent)
                and sse.event == ServerSentEventType._EOF.value
            ):
                return
            yield sse


class JobManager:
    """Very small in-memory job registry (single-process only)."""

    _jobs: ClassVar[Dict[str, JobInfo]] = {}

    def create_job(self) -> JobInfo:
        job_id = uuid.uuid4().hex
        job = JobInfo(id=job_id)
        self._jobs[job_id] = job
        return job

    def get_job(self, job_id: str) -> Optional[JobInfo]:
        return self._jobs.get(job_id)

    @staticmethod
    def mark_done(job: JobInfo, *, error: Optional[str] = None) -> None:
        """Mark a job as done.

        Args:
            job: The JobInfo instance to mark as done.
            error: Optional error message if the job failed.
        """
        job.error = error
        eof = JobStatusEvent.eof()
        job._queue.put_nowait(eof)


job_manager = JobManager()
