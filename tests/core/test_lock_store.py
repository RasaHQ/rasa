import asyncio
from typing import Union
from unittest.mock import patch

import numpy as np
import pytest
import time
from _pytest.tmpdir import TempdirFactory

import rasa.utils.io
from rasa.core.agent import Agent
from rasa.core.channels import UserMessage
from rasa.core.constants import INTENT_MESSAGE_PREFIX
from rasa.core.lock import TicketLock
from rasa.core.lock_store import InMemoryLockStore


@pytest.fixture(scope="session")
def loop():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop = rasa.utils.io.enable_async_loop_debugging(loop)
    yield loop
    loop.close()


def test_issue_ticket():
    lock = TicketLock("random id 0")

    # no lock issued
    assert lock.last_issued == -1
    assert lock.now_serving == 0

    # no one is waiting
    assert not lock.is_someone_waiting()

    # issue ticket
    ticket = lock.issue_ticket(1)
    assert ticket == 0
    assert lock.last_issued == 0
    assert lock.now_serving == 0

    # someone is waiting
    assert lock.is_someone_waiting()


def test_remove_expired_tickets():
    lock = TicketLock("random id 1")

    # issue one long- and one short-lived ticket
    _ = list(map(lock.issue_ticket, [k for k in [0.01, 10]]))

    # both tickets are there
    assert len(lock.tickets) == 2

    # sleep and only one ticket should be left
    time.sleep(0.02)
    lock.remove_expired_tickets()
    assert len(lock.tickets) == 1


def test_create_lock_store():
    lock_store = InMemoryLockStore()
    conversation_id = "my id 0"

    # create lock
    lock_store.create_lock(conversation_id)
    lock = lock_store.get_lock(conversation_id)
    assert lock
    assert lock.conversation_id == conversation_id


def test_serve_ticket():
    lock_store = InMemoryLockStore()
    conversation_id = "my id 1"

    lock = lock_store.create_lock(conversation_id)

    # issue ticket with long lifetime
    ticket_0 = lock.issue_ticket(10)
    assert ticket_0 == 0
    assert lock.last_issued == ticket_0
    assert lock.now_serving == ticket_0
    assert lock.is_someone_waiting()

    # issue another ticket
    ticket_1 = lock.issue_ticket(10)

    # finish serving ticket_0
    lock_store.finish_serving(conversation_id, ticket_0)
    lock = lock_store.get_lock(conversation_id)
    assert lock.last_issued == ticket_1
    assert lock.now_serving == ticket_1
    assert lock.is_someone_waiting()

    # serve second ticket and no one should be waiting
    lock_store.finish_serving(conversation_id, ticket_1)
    assert not lock.is_someone_waiting()


def test_lock_expiration():
    lock_store = InMemoryLockStore()
    conversation_id = "my id 2"
    lock = lock_store.create_lock(conversation_id)

    # issue ticket with long lifetime
    ticket = lock.issue_ticket(10)
    assert ticket == 0
    assert not lock.has_lock_expired(ticket)

    # issue ticket with short lifetime
    ticket = lock.issue_ticket(0.00001)
    time.sleep(0.00002)
    assert ticket == 1
    assert lock.has_lock_expired(ticket)

    # newly assigned ticket should get number 1 again
    assert lock.issue_ticket(10) == 1


async def test_multiple_conversation_ids(default_agent: Agent):
    text = INTENT_MESSAGE_PREFIX + 'greet{"name":"Rasa"}'

    conversation_ids = ["conversation {}".format(i) for i in range(2)]

    # ensure conversations are processed in order
    tasks = [default_agent.handle_text(text, sender_id=_id) for _id in conversation_ids]
    results = await asyncio.gather(*tasks)

    assert results
    processed_ids = [result[0]["recipient_id"] for result in results]
    assert processed_ids == conversation_ids


async def test_message_order(tmpdir_factory: TempdirFactory, default_agent: Agent):
    start_time = time.time()
    n_messages = 10
    lock_wait = 0.1
    # let's write the incoming order and the order of results temp files
    temp_path = tmpdir_factory.mktemp("message_order")
    results_file = temp_path / "results_file.txt"
    incoming_order_file = temp_path / "incoming_order_file.txt"

    # we need to mock `Agent.handle_message()` so we can introduce an
    # artificial holdup (`wait`)
    # We'll also return the message text straight away instead
    async def mocked_handle_message(
        self, message: UserMessage, wait: Union[int, float]
    ) -> None:
        ticket = self.lock_store.issue_ticket(message.sender_id)
        with open(str(incoming_order_file), "a+") as f:
            f.write(message.text + "\n")
        try:
            async with await self.lock_store.lock(
                message.sender_id, ticket, wait=lock_wait
            ):
                await asyncio.sleep(wait)
                with open(str(results_file), "a+") as f:
                    f.write(message.text + "\n")
                return None
        finally:
            self.lock_store.cleanup(message.sender_id, ticket)

    # We'll send n_messages from the same sender_id with different blocking times
    # after the lock has been acquired.
    # We have to ensure that the messages are processed in the right order.
    with patch.object(Agent, "handle_message", mocked_handle_message):
        # use decreasing wait times so that every message after the first one
        # does not acquire its lock immediately
        wait_times = np.linspace(0.1, 0.05, n_messages)
        tasks = [
            default_agent.handle_message(
                UserMessage("sender {0}".format(i), sender_id="some id"), wait=k
            )
            for i, k in enumerate(wait_times)
        ]

        # execute futures
        await asyncio.gather(*(asyncio.ensure_future(t) for t in tasks))

        expected_order = ["sender {0}".format(i) for i in range(len(wait_times))]

        # ensure order of incoming messages is as expected
        with open(str(incoming_order_file)) as f:
            incoming_order = [l for l in f.read().split("\n") if l]
            assert incoming_order == expected_order

        # ensure results are processed in expected order
        with open(str(results_file)) as f:
            results_order = [l for l in f.read().split("\n") if l]
            assert results_order == expected_order

        # Every message after the first one will wait `lock_wait` seconds to acquire its
        # lock (`wait` kwarg in `lock_store.lock()`). Let's make sure that this is not
        # blocking and test that total test execution time is less than
        # the sum of all wait times and (n_messages - 1) * `lock_wait`
        time_limit = np.sum(wait_times[1:])
        time_limit += (n_messages - 1) * lock_wait

        assert time.time() - start_time < time_limit
