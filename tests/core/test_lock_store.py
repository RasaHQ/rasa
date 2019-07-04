import asyncio
import concurrent
from concurrent.futures import ProcessPoolExecutor

from itertools import chain
from typing import Optional, List, Dict, Text, Any, Callable

import numpy as np
import time

import pytest

from unittest.mock import patch

import rasa.utils.io
from rasa.core.agent import Agent
from rasa.core.channels import UserMessage, CollectingOutputChannel
from rasa.core.constants import INTENT_MESSAGE_PREFIX
from rasa.core.domain import Domain
from rasa.core.lock import TicketLock
from rasa.core.lock_store import InMemoryLockStore
from rasa.core.processor import MessageProcessor

domain = Domain.load("data/test_domains/default.yml")


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
    list(map(lock.issue_ticket, [k for k in [0.00001, 10]]))

    # both tickets are there
    assert len(lock.tickets) == 2

    # sleep and only one ticket should be left
    time.sleep(0.00002)
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


async def test_message_order(default_agent: Agent):
    # we need to mock the `Agent.handle_message()` so we can introduce an
    # artificial holdup (`wait`)
    async def mocked_handle_message(
        self,
        message: UserMessage,
        wait: int,
        message_preprocessor: Optional[Callable[[Text], Text]] = None,
        **kwargs
    ) -> Optional[List[Dict[Text, Any]]]:
        if not isinstance(message, UserMessage):
            return await self.handle_text(
                message, message_preprocessor=message_preprocessor, **kwargs
            )

        processor = self.create_processor(message_preprocessor)

        ticket = self.lock_store.issue_ticket(message.sender_id)
        print ("in wait", wait, ticket, message.text)
        try:
            async with await self.lock_store.lock(message.sender_id, ticket, wait=0.02):
                await asyncio.sleep(wait)
                return await processor.handle_message(message)
        finally:
            self.lock_store.cleanup(message.sender_id, ticket)

    # We'll send 10 messages from the same sender_id with different blocking times
    # after the lock has been acquired.
    # We have to ensure that the messages are processed in the right order.
    # Let's repeat the whole test 10 times to rule out randomly correct results.
    with patch.object(Agent, "handle_message", mocked_handle_message):
        wait_times = np.linspace(0.01, 0.001, 10)
        message_tasks = [
            default_agent.handle_message(
                UserMessage(
                    '/greet{{"name":"sender {sender}"}}'.format(sender=i),
                    sender_id="some id",
                ),
                wait=k,
                sender_id="some id",
            )
            for i, k in enumerate(wait_times)
        ]
        restart_tasks = [
            default_agent.handle_message(
                UserMessage("/restart", sender_id="some id"),
                wait=0,
                sender_id="some id",
            )
            for _ in range(len(wait_times))
        ]

        tasks = list(chain.from_iterable(zip(message_tasks, restart_tasks)))

        for t in tasks:
            r = asyncio.ensure_future(t)
            print ("have r", r)
