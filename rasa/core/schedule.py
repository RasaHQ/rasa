import asyncio
import logging

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from pytz import UnknownTimeZoneError, utc

logger = logging.getLogger(__name__)


class ScheduleProvider:

    __scheduler = None

    @staticmethod
    async def get_scheduler() -> AsyncIOScheduler:
        """Thread global scheduler to handle all recurring tasks.

        If no scheduler exists yet, this will instantiate one."""

        if not ScheduleProvider.__scheduler:
            try:
                ScheduleProvider.__scheduler = AsyncIOScheduler(event_loop=asyncio.get_event_loop())
                ScheduleProvider.__scheduler.start()
                return ScheduleProvider.__scheduler
            except UnknownTimeZoneError:
                logger.warning(
                    "apscheduler could not find a timezone and is "
                    "defaulting to utc. This is probably because "
                    "your system timezone is not set. "
                    'Set it with e.g. echo "Europe/Berlin" > '
                    "/etc/timezone"
                )
                ScheduleProvider.__scheduler = AsyncIOScheduler(
                    event_loop=asyncio.get_event_loop(), timezone=utc
                )
                ScheduleProvider.__scheduler.start()
                return ScheduleProvider.__scheduler
        else:
            # scheduler already created, make sure it is running on
            # the correct loop
            # noinspection PyProtectedMember
            if not ScheduleProvider.__scheduler._eventloop == asyncio.get_event_loop():
                raise RuntimeError(
                    "Detected inconsistent loop usage. "
                    "Trying to schedule a task on a new event "
                    "loop, but scheduler was created with a "
                    "different event loop. Make sure there "
                    "is only one event loop in use and that the "
                    "scheduler is running on that one."
                )
            return ScheduleProvider.__scheduler

    @staticmethod
    def stop_scheduler():
        """Terminate the scheduler if started.

        Another call to `scheduler` will create a new scheduler."""

        if ScheduleProvider.__scheduler:
            ScheduleProvider.__scheduler.shutdown()
            ScheduleProvider.__scheduler = None
