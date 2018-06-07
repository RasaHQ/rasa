.. _brokers:

Event Brokers
=============

Rasa Core allows you to stream events to a message broker. The
example implementation we're going to show you here uses `Pika <pika.readthedocs.io>`_,
the Python client library for `RabbitMQ <https://www.rabbitmq.com>`_.

The event broker emits events into the event queue. It becomes part of the
``TrackerStore`` which you use when starting an ``Agent`` or launch
``rasa_core.server``. Here's how you add it:

.. code-block:: python

    from rasa_core.broker import PikaProducer
    from rasa_platform.core.tracker_store import InMemoryTrackerStore

    pika_broker = PikaProducer('localhost',
                                'username',
                                'password',
                                queue='my_events')

    tracker_store = InMemoryTrackerStore(db=db, event_broker=pika_broker)

These events are streamed to RabbitMQ as serialised dictionaries every time
the tracker updates it state. An example event emitted from the ``default``
tracker looks like this:

.. code-block:: json

    {
        "sender_id": "default",
        "timestamp": 1528402837.617099,
        "event": "bot"
    }

The ``event`` field takes the event's ``type_name`` (for more on event
types, check out the :doc:`api/events` docs). You need to have a RabbitMQ
server running, as well as another application
that consumes the events. This consumer to needs to implement Pika's
``start_consuming()`` method with a ``callback`` action. Here's a simple
example:

.. code-block:: python

    import json
    import pika

    class MyConsumer(object):

        def __init__(self, host, username, password, queue='my_events'):
            credentials = pika.PlainCredentials(username, password)
            connection = pika.BlockingConnection(pika.ConnectionParameters(host, credentials=credentials))
            self.queue = queue
            self.channel = connection.channel()

        def consume(self):
            self.channel.basic_consume(self._callback,
                                       queue=self.queue,
                                       no_ack=True)
            self.channel.start_consuming()

        def _callback(self, ch, method, properties, body):
            # Do something useful with your incoming message body here, e.g.
            # saving it to a database
            print('Received event {}'.format(json.loads(body)))

    if __name__ == '__main__':
        consumer = MyConsumer('localhost', 'username', 'password')
        consumer.consume()





