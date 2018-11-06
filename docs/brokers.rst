.. _brokers:

Event Brokers
=============

Rasa Core allows you to stream events to a message broker. The
example implementation we're going to show you here uses `Pika <pika.readthedocs.io>`_,
the Python client library for `RabbitMQ <https://www.rabbitmq.com>`_.

The event broker emits events into the event queue. It becomes part of the
``TrackerStore`` which you use when starting an ``Agent`` or launch
``rasa_core.run``.

Adding an Event Broker Using the Endpoint Configuration
-------------------------------------------------------

You can use an endpoint configuration file to instruct Rasa Core to stream
all events to your event broker. To do so, add the following section to your
endpoint configuration, e.g. ``endpoints.yml``:

.. literalinclude:: ../data/test_endpoints/event_broker_endpoint.yml

Then instruct Rasa Core to use the endpoint configuration by adding
``--endpoints <path to your endpoint configuration`` when running it.

Adding an Event Broker in Python
--------------------------------

Here is how you add it using Python code:

.. code-block:: python

    from rasa_core.broker import PikaProducer
    from rasa_platform.core.tracker_store import InMemoryTrackerStore

    pika_broker = PikaProducer('localhost',
                                'username',
                                'password',
                                queue='rasa_core_events')

    tracker_store = InMemoryTrackerStore(db=db, event_broker=pika_broker)


Implementing an Event Consumer
------------------------------

All events are streamed to RabbitMQ as serialised dictionaries every time
the tracker updates it state. An example event emitted from the ``default``
tracker looks like this:

.. code-block:: json

    {
        "sender_id": "default",
        "timestamp": 1528402837.617099,
        "event": "bot",
        "text": "what your bot said",
        "data": "some data"
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


    def _callback(self, ch, method, properties, body):
            # Do something useful with your incoming message body here, e.g.
            # saving it to a database
            print('Received event {}'.format(json.loads(body)))

    if __name__ == '__main__':

        # RabbitMQ credentials with username and password
        credentials = pika.PlainCredentials('username', 'password')

        # pika connection to the RabbitMQ host - typically 'rabbit' in a
        # docker environment, or 'localhost' in a local environment
        connection = pika.BlockingConnection(
            pika.ConnectionParameters('rabbit', credentials=credentials))

        # start consumption of channel
        channel = connection.channel()
        channel.basic_consume(_callback,
                              queue='rasa_core_events',
                              no_ack=True)
        channel.start_consuming()


.. include:: feedback.inc


