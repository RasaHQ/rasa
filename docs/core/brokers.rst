:desc: Find out how open source chatbot framework Rasa Stack allows
       you to stream events to a message broker.

.. _brokers:

Event Brokers
=============

Rasa Core allows you to stream events to a message broker. The event broker
emits events into the event queue. It becomes part of the ``TrackerStore``
which you use when starting an ``Agent`` or launch ``rasa.core.run``.

All events are streamed to the broker as serialised dictionaries every time
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
types, check out the :doc:`api/events` docs).

Rasa enables two possible brokers producers: Pika Event Broker and Kafka Event Broker.

Pika Event Broker
-----------------

The example implementation we're going to show you here uses `Pika <pika.readthedocs.io>`_,
the Python client library for `RabbitMQ <https://www.rabbitmq.com>`_.

Adding a Pika Event Broker Using the Endpoint Configuration
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

You can use an endpoint configuration file to instruct Rasa Core to stream
all events to your event broker. To do so, add the following section to your
endpoint configuration, e.g. ``endpoints.yml``:

.. literalinclude:: ../../data/test_endpoints/event_brokers/pika_endpoint.yml

Then instruct Rasa Core to use the endpoint configuration and Pika producer by adding
``--endpoints <path to your endpoint configuration`` as following example:

.. code-block:: shell

    rasa run -m models --endpoints endpoints.yml

Adding a Pika Event Broker in Python
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Here is how you add it using Python code:

.. code-block:: python

    from rasa.core.event_brokers.pika_producer import PikaProducer
    from rasa_platform.core.tracker_store import InMemoryTrackerStore

    pika_broker = PikaProducer('localhost',
                                'username',
                                'password',
                                queue='rasa_core_events')

    tracker_store = InMemoryTrackerStore(db=db, event_broker=pika_broker)


Implementing a Pika Event Consumer
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

You need to have a RabbitMQ server running, as well as another application
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

Kafka Event Broker
------------------

It is possible to use `Kafka <https://kafka.apache.org/>`_ as main broker to you events. In this example
we are going to use the `python-kafka <https://kafka-python.readthedocs.io/en/master/usage.html>`_
library, a Kafka client written in Python.

Adding a Kafka Event Broker Using the Endpoint Configuration
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

As for the other brokers, you can use an endpoint configuration file to instruct Rasa Core to stream
all events to this event broker. To do it, add the following section to your
endpoint configuration.

Pass the ``endpoints.yml`` file as argument with ``--endpoints <path to your endpoint configuration>``
when running Rasa, as following example:

.. code-block:: shell

    rasa run -m models --endpoints endpoints.yml

Using ``SASL_PLAINTEXT`` protocol the endpoints file must have the following entries:

.. literalinclude:: ../../data/test_endpoints/event_brokers/kafka_plaintext_endpoint.yml

In the case of using SSL protocol the endpoints file must looks like:

.. literalinclude:: ../../data/test_endpoints/event_brokers/kafka_ssl_endpoint.yml

Adding a Kafka Broker in Python
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The code below shows an example on how to instantiate a Kafka producer in you script.

.. code-block:: python

    from rasa.core.event_brokers.kafka_producer import KafkaProducer
    from rasa.core.tracker_store import InMemoryTrackerStore

    kafka_broker = KafkaProducer(host='localhost:9092',
                                 topic='rasa_core_events')

    tracker_store = InMemoryTrackerStore(event_broker=kafka_broker)


The host variable can be either a list of brokers adresses or a single one.
If only one broker address is available, the client will connect to it and 
request the cluster Metadata.
Therefore, the remain brokers in the cluster can be discovered
automatically through the data served by the first connected broker.

To pass more than one broker address as argument, they must be passed in a
list of strings. e.g.:

.. code-block:: python

    kafka_broker = KafkaProducer(host=['kafka_broker_1:9092',
                                       'kafka_broker_2:2030',
                                       'kafka_broker_3:9092'],
                                 topic='rasa_core_events')

Authentication and authorization
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Rasa Core's Kafka producer accepts two types of security protocols - ``SASL_PLAINTEXT`` and ``SSL``.

For development environment, or if the brokers servers and clients are located
into the same machine, you can use simple authentication with ``SASL_PLAINTEXT``.
By using this protocol, the credentials and messages exchanged between the clients and servers
will be sent in plaintext. Thus, this is not the most secure approach, but since it's simple
to configure, it is useful for simple cluster configurations.
``SASL_PLAINTEXT`` protocol requires the setup of the ``username`` and ``password``
previously configured in the broker server.

.. code-block:: python

    kafka_broker = KafkaProducer(host='kafka_broker:9092',
                                 sasl_plain_username='kafka_username',
                                 sasl_plain_password='kafka_password',
                                 security_protocol='SASL_PLAINTEXT',
                                 topic='rasa_core_events')


If the clients or the brokers in the kafka cluster are located in different
machines, it's important to use ssl protocal to assure encryption of data and client
authentication. After generating valid certificates for the brokers and the
clients, the path to the certificate and key generated for the producer must
be provided as arguments, as well as the CA's root certificate.

.. code-block:: python

    kafka_broker = KafkaProducer(host='kafka_broker:9092',
                                 ssl_cafile='CARoot.pem',
                                 ssl_certfile='certificate.pem',
                                 ssl_keyfile='key.pem',
                                 ssl_check_hostname=True,
                                 security_protocol='SSL',
                                 topic='rasa_core_events')

If the ``ssl_check_hostname`` parameter is enabled, the clients will verify
if the broker's hostname matches the certificate. It's used on client's connections
and inter-broker connections to prevent man-in-the-middle attacks.


Implementing a Kafka Event Consumer
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The parameters used to create a Kafka consumer is the same used on the producer creation,
according to the security protocol being used. The following implementation shows an example:

.. code-block:: python

    from kafka import KafkaConsumer
    from json import loads

    consumer = KafkaConsumer('rasa_core_events',
                              bootstrap_servers=['localhost:29093'],
                              value_deserializer=lambda m: json.loads(m.decode('utf-8')),
                              security_protocol='SSL',
                              ssl_check_hostname=False,
                              ssl_cafile='CARoot.pem',
                              ssl_certfile='certificate.pem',
                              ssl_keyfile='key.pem')

    for message in consumer:
        print(message.value)

.. include:: feedback.inc
