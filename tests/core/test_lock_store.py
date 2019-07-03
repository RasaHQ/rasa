import asyncio
import time

import pytest

import rasa.utils.io
from rasa.core.agent import Agent
from rasa.core.constants import INTENT_MESSAGE_PREFIX
from rasa.core.domain import Domain
from rasa.core.lock import TicketLock
from rasa.core.lock_store import InMemoryLockStore

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

    # issue another ticket
    ticket_1 = lock.issue_ticket(10)

    # finish serving ticket_0
    lock_store.finish_serving(conversation_id, ticket_0)
    lock = lock_store.get_lock(conversation_id)
    assert lock.now_serving == ticket_1
    assert lock.now_serving == ticket_1


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


# async def test_expired_locks(default_agent: Agent):
#     default_agent.lock_store.lifetime = 0
#     text = INTENT_MESSAGE_PREFIX + 'greet{"name":"Rasa"}'
#
#     result = await default_agent.handle_text(text, sender_id="some id")
#
#     assert result
