import asyncio
import logging
import multiprocessing.pool
import os
import time
from collections import deque
from pathlib import Path
from typing import Iterator

import pytest
from _pytest.logging import LogCaptureFixture
from rasa.core.channels import UserMessage
from rasa.core.lock_store import LockStore
from rasa.utils.endpoints import EndpointConfig

from rasa.core.concurrent_lock_store import (
    LAST_ISSUED_TICKET_NUMBER_SUFFIX,
    ConcurrentRedisLockStore,
)

REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = os.getenv("REDIS_PORT", 6380)


@pytest.fixture
def concurrent_redis_lock_store() -> Iterator[ConcurrentRedisLockStore]:
    # we need one redis database per worker, otherwise
    # tests conflicts with each others when databases are flushed
    pytest_worker_id = os.getenv("PYTEST_XDIST_WORKER", "gw0")
    redis_database = int(pytest_worker_id.replace("gw", ""))
    lock_store = ConcurrentRedisLockStore(
        EndpointConfig(
            host=REDIS_HOST,
            port=REDIS_PORT,
            db=redis_database,
        )
    )
    try:
        yield lock_store
    finally:
        lock_store.red.flushdb()


class FakeAgent:
    def __init__(self, lock_store: LockStore, results_file: Path) -> None:
        self.results_file = results_file
        self.lock_store = lock_store

    async def mocked_handle_message(self, message: UserMessage) -> None:
        lock_wait = 0.5
        async with self.lock_store.lock(
            message.sender_id, wait_time_in_seconds=lock_wait
        ):
            # write message to file as it's processed
            with open(str(self.results_file), "a+") as f_1:
                f_1.write(message.text + "\n")

            return None


async def handle_message(
    lock_store: LockStore, message: UserMessage, results_file: Path
) -> None:
    agent = FakeAgent(lock_store, results_file)
    return await agent.mocked_handle_message(message)


@pytest.mark.concurrent_lock_store
async def test_concurrent_message_handling(
    tmp_path: Path,
    concurrent_redis_lock_store: ConcurrentRedisLockStore,
) -> None:
    results_file = tmp_path / "results_file"

    message_text = "This is a test."
    user_message = UserMessage(message_text, sender_id="some id")

    pool = multiprocessing.pool.ThreadPool(2)
    result = pool.starmap_async(
        handle_message,
        [
            (concurrent_redis_lock_store, user_message, results_file),
            (concurrent_redis_lock_store, user_message, results_file),
        ],
    )
    pool.close()
    pool.join()

    await asyncio.gather(*(coro for coro in result.get(timeout=120)))

    expected_order = [message_text, message_text]

    # ensure results are as expected
    with open(str(results_file)) as f:
        results_order = [line for line in f.read().split("\n") if line]
        assert results_order == expected_order


@pytest.mark.concurrent_lock_store
def test_create_concurrent_lock_store(
    concurrent_redis_lock_store: ConcurrentRedisLockStore,
) -> None:
    conversation_id = "my id 0"

    ticket_number = concurrent_redis_lock_store.issue_ticket(conversation_id)
    assert ticket_number == 1
    lock = concurrent_redis_lock_store.get_lock(conversation_id)
    assert lock
    assert lock.conversation_id == conversation_id


@pytest.mark.concurrent_lock_store
def test_concurrent_serve_ticket(
    concurrent_redis_lock_store: ConcurrentRedisLockStore,
) -> None:
    conversation_id = "my id 1"

    # issue ticket with long lifetime
    ticket_0 = concurrent_redis_lock_store.issue_ticket(conversation_id, 10)
    assert ticket_0 == 1

    lock = concurrent_redis_lock_store.get_lock(conversation_id)

    assert lock is not None
    assert lock.last_issued == ticket_0
    assert lock.now_serving == ticket_0
    assert lock.is_someone_waiting()

    # issue other tickets
    ticket_1 = concurrent_redis_lock_store.issue_ticket(conversation_id, 10)
    ticket_2 = concurrent_redis_lock_store.issue_ticket(conversation_id, 10)

    # finish serving ticket_0
    concurrent_redis_lock_store.finish_serving(conversation_id, ticket_0)

    lock = concurrent_redis_lock_store.get_lock(conversation_id)

    assert lock.last_issued == ticket_2
    assert lock.now_serving == ticket_1
    assert lock.is_someone_waiting()

    # # serve second and third ticket and no one should be waiting
    concurrent_redis_lock_store.finish_serving(conversation_id, ticket_1)
    concurrent_redis_lock_store.finish_serving(conversation_id, ticket_2)

    lock = concurrent_redis_lock_store.get_lock(conversation_id)
    assert not lock.is_someone_waiting()


# noinspection PyProtectedMember
@pytest.mark.concurrent_lock_store
def test_concurrent_lock_expiration(
    concurrent_redis_lock_store: ConcurrentRedisLockStore,
) -> None:
    conversation_id = "my id 2"
    initial_ticket_number = 1

    # issue ticket with long lifetime
    ticket_one = concurrent_redis_lock_store.issue_ticket(conversation_id, 10)
    assert ticket_one == initial_ticket_number
    lock = concurrent_redis_lock_store.get_lock(conversation_id)
    assert lock is not None
    assert not lock._ticket_for_ticket_number(ticket_one).has_expired()

    # issue ticket with short lifetime
    ticket_two = concurrent_redis_lock_store.issue_ticket(conversation_id, 0.00001)
    time.sleep(0.00002)
    assert ticket_two == initial_ticket_number + 1
    lock = concurrent_redis_lock_store.get_lock(conversation_id)
    assert lock is not None
    assert lock._ticket_for_ticket_number(ticket_two) is None

    # newly assigned ticket should increment once more, regardless of ticket expiring
    ticket_three = concurrent_redis_lock_store.issue_ticket(conversation_id, 10)
    assert ticket_three == initial_ticket_number + 2


@pytest.mark.concurrent_lock_store
def test_concurrent_get_lock(
    concurrent_redis_lock_store: ConcurrentRedisLockStore,
) -> None:
    conversation_id = "my id 3"

    lock = concurrent_redis_lock_store.get_lock(conversation_id)
    assert lock is not None
    assert lock.tickets == deque()

    # issue several tickets
    for _ in range(5):
        concurrent_redis_lock_store.issue_ticket(conversation_id, 20)

    lock = concurrent_redis_lock_store.get_lock(conversation_id)

    assert len(lock.tickets) == 5

    for i in range(5):
        # ticket numbers start at 1, not 0
        assert lock.tickets[i].number == i + 1


@pytest.mark.concurrent_lock_store
def test_concurrent_delete_lock_success(
    concurrent_redis_lock_store: ConcurrentRedisLockStore,
    caplog: LogCaptureFixture,
) -> None:
    conversation_id = "my id 4"

    # issue several tickets
    for _ in range(4):
        concurrent_redis_lock_store.issue_ticket(conversation_id, 20)

    with caplog.at_level(logging.DEBUG):
        concurrent_redis_lock_store.delete_lock(conversation_id)

    assert f"Deleted lock for conversation '{conversation_id}'." in caplog.text

    lock = concurrent_redis_lock_store.get_lock(conversation_id)
    assert lock is not None
    assert len(lock.tickets) == 0


@pytest.mark.concurrent_lock_store
def test_concurrent_delete_lock_no_keys(
    concurrent_redis_lock_store: ConcurrentRedisLockStore,
    caplog: LogCaptureFixture,
) -> None:
    conversation_id = "some id"

    with caplog.at_level(logging.DEBUG):
        concurrent_redis_lock_store.delete_lock(conversation_id)

    assert (
        f"The lock store does not contain any key-value items "
        f"for conversation '{conversation_id}'." in caplog.text
    )


@pytest.mark.concurrent_lock_store
def test_concurrent_increment_ticket_number_and_save_lock(
    concurrent_redis_lock_store: ConcurrentRedisLockStore,
) -> None:
    conversation_id = "my id 5"
    lock = concurrent_redis_lock_store.get_lock(conversation_id)
    assert lock is not None
    assert len(lock.tickets) == 0

    total_issued_tickets = 3

    # issue several tickets
    for _ in range(total_issued_tickets):
        ticket_number = concurrent_redis_lock_store.increment_ticket_number(lock)
        lock.concurrent_issue_ticket(lifetime=100, ticket_number=ticket_number)
        concurrent_redis_lock_store.save_lock(lock)

    last_issued_key = (
        concurrent_redis_lock_store.key_prefix
        + conversation_id
        + ":"
        + LAST_ISSUED_TICKET_NUMBER_SUFFIX
    )
    retrieved_key = concurrent_redis_lock_store.red.get(last_issued_key)
    assert retrieved_key is not None
    assert int(retrieved_key) == total_issued_tickets

    retrieved_lock = concurrent_redis_lock_store.get_lock(conversation_id)
    assert retrieved_lock is not None
    for j in range(3):
        # ticket numbers start at 1
        assert retrieved_lock.tickets[j].number == j + 1
        assert retrieved_lock.tickets[j].expires


@pytest.mark.concurrent_lock_store
def test_concurrent_finish_serving(
    concurrent_redis_lock_store: ConcurrentRedisLockStore,
) -> None:
    conversation_id = "my id 6"

    # issue several tickets
    for _ in range(5):
        concurrent_redis_lock_store.issue_ticket(conversation_id)

    expected_last_issued = 5

    last_issued_key = (
        concurrent_redis_lock_store.key_prefix
        + conversation_id
        + ":"
        + LAST_ISSUED_TICKET_NUMBER_SUFFIX
    )
    retrieved_key = concurrent_redis_lock_store.red.get(last_issued_key)
    assert retrieved_key is not None
    assert int(retrieved_key) == expected_last_issued

    # finish serving last ticket
    concurrent_redis_lock_store.finish_serving(
        conversation_id, ticket_number=expected_last_issued
    )

    lock = concurrent_redis_lock_store.get_lock(conversation_id)
    assert lock is not None
    assert lock.last_issued == 4

    # issue a new ticket
    concurrent_redis_lock_store.issue_ticket(conversation_id)
    retrieved_key = concurrent_redis_lock_store.red.get(last_issued_key)
    assert retrieved_key is not None
    assert int(retrieved_key) == expected_last_issued + 1
