import docker

from rasa.core.brokers.celery import CeleryEventBroker

from .conftest import REDIS_HOST, REDIS_PORT


async def test_celery_send_task():
    broker = CeleryEventBroker(broker_url="redis://localhost:6379/0", task_name="event")
    arg = {"event": "data"}
    broker.publish(arg)
    for worker, tasks in broker.active().items():
        assert tasks[0] == arg
