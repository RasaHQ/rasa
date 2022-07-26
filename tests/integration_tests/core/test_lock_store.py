import asyncio
import logging
import multiprocessing
import time
from collections import deque
from pathlib import Path

import numpy as np
import pytest
from _pytest.logging import LogCaptureFixture
from _pytest.monkeypatch import MonkeyPatch

from rasa.core.agent import Agent
from rasa.core.channels import UserMessage
from rasa.shared.constants import INTENT_MESSAGE_PREFIX
from rasa.core.lock_store import (
    ConcurrentRedisLockStore,
    LockError,
    RedisLockStore,
    LAST_ISSUED_TICKET_NUMBER_SUFFIX,
)


def test_create_lock_store(redis_lock_store: RedisLockStore):
    conversation_id = "my id 0"

    # create and lock
    lock = redis_lock_store.create_lock(conversation_id)
    redis_lock_store.save_lock(lock)
    lock = redis_lock_store.get_lock(conversation_id)
    assert lock
    assert lock.conversation_id == conversation_id


def test_serve_ticket(redis_lock_store: RedisLockStore):
    conversation_id = "my id 1"

    lock = redis_lock_store.create_lock(conversation_id)
    redis_lock_store.save_lock(lock)

    # issue ticket with long lifetime
    ticket_0 = redis_lock_store.issue_ticket(conversation_id, 10)
    assert ticket_0 == 0

    lock = redis_lock_store.get_lock(conversation_id)
    assert lock.last_issued == ticket_0
    assert lock.now_serving == ticket_0
    assert lock.is_someone_waiting()

    # issue another ticket
    ticket_1 = redis_lock_store.issue_ticket(conversation_id, 10)

    # finish serving ticket_0
    redis_lock_store.finish_serving(conversation_id, ticket_0)

    lock = redis_lock_store.get_lock(conversation_id)

    assert lock.last_issued == ticket_1
    assert lock.now_serving == ticket_1
    assert lock.is_someone_waiting()

    # serve second ticket and no one should be waiting
    redis_lock_store.finish_serving(conversation_id, ticket_1)

    lock = redis_lock_store.get_lock(conversation_id)
    assert not lock.is_someone_waiting()


# noinspection PyProtectedMember
def test_lock_expiration(redis_lock_store: RedisLockStore):
    conversation_id = "my id 2"
    lock = redis_lock_store.create_lock(conversation_id)
    redis_lock_store.save_lock(lock)

    # issue ticket with long lifetime
    ticket_number = redis_lock_store.increment_ticket_number(lock)
    ticket = lock.issue_ticket(10, ticket_number)
    assert ticket == 0
    assert not lock._ticket_for_ticket_number(ticket).has_expired()

    # issue ticket with short lifetime
    ticket_number = redis_lock_store.increment_ticket_number(lock)
    ticket = lock.issue_ticket(0.00001, ticket_number)
    time.sleep(0.00002)
    assert ticket == 1
    assert lock._ticket_for_ticket_number(ticket) is None

    # newly assigned ticket should get number 1 again
    ticket_number = redis_lock_store.increment_ticket_number(lock)
    assert lock.issue_ticket(10, ticket_number) == 1


async def test_multiple_conversation_ids(default_agent: Agent):
    text = INTENT_MESSAGE_PREFIX + 'greet{"name":"Rasa"}'

    conversation_ids = [f"conversation {i}" for i in range(2)]

    # ensure conversations are processed in order
    tasks = [default_agent.handle_text(text, sender_id=_id) for _id in conversation_ids]
    results = await asyncio.gather(*tasks)

    assert results
    processed_ids = [result[0]["recipient_id"] for result in results]
    assert processed_ids == conversation_ids


async def test_message_order(
    tmp_path: Path, default_agent: Agent, monkeypatch: MonkeyPatch
):
    start_time = time.time()
    n_messages = 10
    lock_wait = 0.5

    # let's write the incoming order of messages and the order of results to temp files
    results_file = tmp_path / "results_file"
    incoming_order_file = tmp_path / "incoming_order_file"

    # We need to mock `Agent.handle_message()` so we can introduce an
    # artificial holdup (`wait_time_in_seconds`). In the mocked method, we'll
    # record messages as they come and and as they're processed in files so we
    # can check the order later on. We don't need the return value of this method so
    # we'll just return None.
    async def mocked_handle_message(self, message: UserMessage, wait: float) -> None:
        # write incoming message to file
        with open(str(incoming_order_file), "a+") as f_0:
            f_0.write(message.text + "\n")

        async with self.lock_store.lock(
            message.sender_id, wait_time_in_seconds=lock_wait
        ):
            # hold up the message processing after the lock has been acquired
            await asyncio.sleep(wait)

            # write message to file as it's processed
            with open(str(results_file), "a+") as f_1:
                f_1.write(message.text + "\n")

            return None

    # We'll send n_messages from the same sender_id with different blocking times
    # after the lock has been acquired.
    # We have to ensure that the messages are processed in the right order.
    monkeypatch.setattr(Agent, "handle_message", mocked_handle_message)
    # use decreasing wait times so that every message after the first one
    # does not acquire its lock immediately
    wait_times = np.linspace(0.1, 0.05, n_messages)
    tasks = [
        default_agent.handle_message(
            UserMessage(f"sender {i}", sender_id="some id"), wait=k
        )
        for i, k in enumerate(wait_times)
    ]

    # execute futures
    await asyncio.gather(*(asyncio.ensure_future(t) for t in tasks))

    expected_order = [f"sender {i}" for i in range(len(wait_times))]

    # ensure order of incoming messages is as expected
    with open(str(incoming_order_file)) as f:
        incoming_order = [line for line in f.read().split("\n") if line]
        assert incoming_order == expected_order

    # ensure results are processed in expected order
    with open(str(results_file)) as f:
        results_order = [line for line in f.read().split("\n") if line]
        assert results_order == expected_order

    # Every message after the first one will wait `lock_wait` seconds to acquire its
    # lock (`wait_time_in_seconds` kwarg in `lock_store.lock()`).
    # Let's make sure that this is not blocking and test that total test
    # execution time is less than  the sum of all wait times plus
    # (n_messages - 1) * lock_wait
    time_limit = np.sum(wait_times[1:])
    time_limit += (n_messages - 1) * lock_wait
    assert time.time() - start_time < time_limit


async def test_lock_error(default_agent: Agent, monkeypatch: MonkeyPatch):
    lock_lifetime = 0.01
    wait_time_in_seconds = 0.01
    holdup = 0.5

    # Mock message handler again to add a wait time holding up the lock
    # after it's been acquired
    async def mocked_handle_message(self, message: UserMessage) -> None:
        async with self.lock_store.lock(
            message.sender_id,
            wait_time_in_seconds=wait_time_in_seconds,
            lock_lifetime=lock_lifetime,
        ):
            # hold up the message processing after the lock has been acquired
            await asyncio.sleep(holdup)

        return None

    monkeypatch.setattr(Agent, "handle_message", mocked_handle_message)
    # first message blocks the lock for `holdup`,
    # meaning the second message will not be able to acquire a lock
    tasks = [
        default_agent.handle_message(UserMessage(f"sender {i}", sender_id="some id"))
        for i in range(2)
    ]

    with pytest.raises(LockError):
        await asyncio.gather(*(asyncio.ensure_future(t) for t in tasks))


class FakeAgent:
    def __init__(self, lock_store, results_file):
        self.results_file = results_file
        self.lock_store = lock_store

    async def mocked_handle_message(self, message) -> None:
        lock_wait = 0.5
        async with self.lock_store.lock(
            message.sender_id, wait_time_in_seconds=lock_wait
        ):
            # write message to file as it's processed
            with open(str(self.results_file), "a+") as f_1:
                f_1.write(message.text + "\n")

            return None


async def handle_message(lock_store, message, results_file):
    agent = FakeAgent(lock_store, results_file)
    return await agent.mocked_handle_message(message)


async def test_concurrent_message_handling(
    tmp_path: Path,
    concurrent_redis_lock_store: ConcurrentRedisLockStore,
):
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


def test_create_concurrent_lock_store(
    concurrent_redis_lock_store: ConcurrentRedisLockStore,
):
    conversation_id = "my id 0"

    ticket_number = concurrent_redis_lock_store.issue_ticket(conversation_id)
    assert ticket_number == 1
    lock = concurrent_redis_lock_store.get_lock(conversation_id)
    assert lock
    assert lock.conversation_id == conversation_id


def test_concurrent_serve_ticket(concurrent_redis_lock_store: ConcurrentRedisLockStore):
    conversation_id = "my id 1"

    # issue ticket with long lifetime
    ticket_0 = concurrent_redis_lock_store.issue_ticket(conversation_id, 10)
    assert ticket_0 == 1

    lock = concurrent_redis_lock_store.get_lock(conversation_id)
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
def test_concurrent_lock_expiration(
    concurrent_redis_lock_store: ConcurrentRedisLockStore,
):
    conversation_id = "my id 2"
    initial_ticket_number = 1

    # issue ticket with long lifetime
    ticket_one = concurrent_redis_lock_store.issue_ticket(conversation_id, 10)
    assert ticket_one == initial_ticket_number
    lock = concurrent_redis_lock_store.get_lock(conversation_id)
    assert not lock._ticket_for_ticket_number(ticket_one).has_expired()

    # issue ticket with short lifetime
    ticket_two = concurrent_redis_lock_store.issue_ticket(conversation_id, 0.00001)
    time.sleep(0.00002)
    assert ticket_two == initial_ticket_number + 1
    lock = concurrent_redis_lock_store.get_lock(conversation_id)
    assert lock._ticket_for_ticket_number(ticket_two) is None

    # newly assigned ticket should increment once more, regardless of ticket expiring
    ticket_three = concurrent_redis_lock_store.issue_ticket(conversation_id, 10)
    assert ticket_three == initial_ticket_number + 2


def test_concurrent_get_lock(concurrent_redis_lock_store: ConcurrentRedisLockStore):
    conversation_id = "my id 3"

    lock = concurrent_redis_lock_store.get_lock(conversation_id)
    assert lock.tickets == deque()

    # issue several tickets
    for _ in range(5):
        concurrent_redis_lock_store.issue_ticket(conversation_id, 20)

    lock = concurrent_redis_lock_store.get_lock(conversation_id)

    assert len(lock.tickets) == 5

    for i in range(5):
        # ticket numbers start at 1, not 0
        assert lock.tickets[i].number == i + 1


def test_concurrent_delete_lock_success(
    concurrent_redis_lock_store: ConcurrentRedisLockStore,
    caplog: LogCaptureFixture,
):
    conversation_id = "my id 4"

    # issue several tickets
    for _ in range(4):
        concurrent_redis_lock_store.issue_ticket(conversation_id, 20)

    with caplog.at_level(logging.DEBUG):
        concurrent_redis_lock_store.delete_lock(conversation_id)

    assert f"Deleted lock for conversation '{conversation_id}'." in caplog.text

    lock = concurrent_redis_lock_store.get_lock(conversation_id)
    assert len(lock.tickets) == 0


def test_concurrent_delete_lock_no_keys(
    concurrent_redis_lock_store: ConcurrentRedisLockStore,
    caplog: LogCaptureFixture,
):
    conversation_id = "some id"

    with caplog.at_level(logging.DEBUG):
        concurrent_redis_lock_store.delete_lock(conversation_id)

    assert (
        f"The lock store does not contain any key-value items "
        f"for conversation '{conversation_id}'." in caplog.text
    )


def test_concurrent_increment_ticket_number_and_save_lock(
    concurrent_redis_lock_store: ConcurrentRedisLockStore,
):
    conversation_id = "my id 5"
    lock = concurrent_redis_lock_store.get_lock(conversation_id)
    assert len(lock.tickets) == 0

    total_issued_tickets = 3

    # issue several tickets
    for _ in range(total_issued_tickets):
        ticket_number = concurrent_redis_lock_store.increment_ticket_number(lock)
        lock.issue_ticket(lifetime=100, ticket_number=ticket_number)
        concurrent_redis_lock_store.save_lock(lock)

    last_issued_key = (
        concurrent_redis_lock_store.key_prefix
        + conversation_id
        + ":"
        + LAST_ISSUED_TICKET_NUMBER_SUFFIX
    )
    assert (
        int(concurrent_redis_lock_store.red.get(last_issued_key))
        == total_issued_tickets
    )

    retrieved_lock = concurrent_redis_lock_store.get_lock(conversation_id)
    for j in range(3):
        # ticket numbers start at 1
        assert retrieved_lock.tickets[j].number == j + 1
        assert retrieved_lock.tickets[j].expires


def test_concurrent_finish_serving(
    concurrent_redis_lock_store: ConcurrentRedisLockStore,
):
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
    assert (
        int(concurrent_redis_lock_store.red.get(last_issued_key))
        == expected_last_issued
    )

    # finish serving last ticket
    concurrent_redis_lock_store.finish_serving(
        conversation_id, ticket_number=expected_last_issued
    )

    lock = concurrent_redis_lock_store.get_lock(conversation_id)
    assert lock.last_issued == 4

    # issue a new ticket
    concurrent_redis_lock_store.issue_ticket(conversation_id)
    assert (
        int(concurrent_redis_lock_store.red.get(last_issued_key))
        == expected_last_issued + 1
    )
