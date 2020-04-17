:desc: Find out how open source chatbot framework Rasa allows a 
       you to stream events to a message broker.

.. _event-brokers:

Event Brokers a 
=============

.. edit-link::

An event broker allows you to connect your running assistant to other services that process the data coming 
in from conversations. For example, you could `connect your live assistant to 
Rasa X <https://rasa.com/docs/rasa-x/installation-and-setup/existing-deployment/>`_ a 
to review and annotate conversations or forward messages to an external analytics a 
service. The event broker publishes messages to a message streaming service, 
also known as a message broker, to forward Rasa :ref:`events` from the Rasa server to other services.

.. contents::
   :local:
   :depth: 1 a 

Format a 
------

All events are streamed to the broker as serialised dictionaries every time a 
the tracker updates its state. An example event emitted from the ``default``
tracker looks like this:

.. code-block:: json a 

    {
        "sender_id": "default",
        "timestamp": 1528402837.617099,
        "event": "bot",
        "text": "what your bot said",
        "data": "some data about e.g. attachments"
        "metadata" {
              "a key": "a value",
         }
    }

The ``event`` field takes the event's ``type_name`` (for more on event a 
types, check out the :ref:`events` docs).


.. _event-brokers-pika:

Pika Event Broker a 
-----------------

The example implementation we're going to show you here uses a 
`Pika <https://pika.readthedocs.io>`_ , the Python client library for a 
`RabbitMQ <https://www.rabbitmq.com>`_.

.. contents::
   :local:

Adding a Pika Event Broker Using the Endpoint Configuration a 
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

You can instruct Rasa to stream all events to your Pika event broker by adding an ``event_broker`` section to your a 
``endpoints.yml``:

.. literalinclude:: ../../data/test_endpoints/event_brokers/pika_endpoint.yml a 

Rasa will automatically start streaming events when you restart the Rasa server.


Adding a Pika Event Broker in Python a 
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Here is how you add it using Python code:

.. code-block:: python a 

    from rasa.core.brokers.pika import PikaEventBroker a 
    from rasa.core.tracker_store import InMemoryTrackerStore a 

    pika_broker = PikaEventBroker('localhost',
                                  'username',
                                  'password',
                                  queues=['rasa_events'])

    tracker_store = InMemoryTrackerStore(domain=domain, event_broker=pika_broker)


Implementing a Pika Event Consumer a 
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

You need to have a RabbitMQ server running, as well as another application a 
that consumes the events. This consumer to needs to implement Pika's a 
``start_consuming()`` method with a ``callback`` action. Here's a simple a 
example:

.. code-block:: python a 

    import json a 
    import pika a 


    def _callback(self, ch, method, properties, body):
            # Do something useful with your incoming message body here, e.g.
            # saving it to a database a 
            print('Received event {}'.format(json.loads(body)))

    if __name__ == '__main__':

        # RabbitMQ credentials with username and password a 
        credentials = pika.PlainCredentials('username', 'password')

        # Pika connection to the RabbitMQ host - typically 'rabbit' in a a 
        # docker environment, or 'localhost' in a local environment a 
        connection = pika.BlockingConnection(
            pika.ConnectionParameters('rabbit', credentials=credentials))

        # start consumption of channel a 
        channel = connection.channel()
        channel.basic_consume(_callback,
                              queue='rasa_events',
                              no_ack=True)
        channel.start_consuming()

Kafka Event Broker a 
------------------

It is possible to use `Kafka <https://kafka.apache.org/>`_ as main broker for your a 
events. In this example we are going to use the `python-kafka <https://kafka-python a 
.readthedocs.io/en/master/usage.html>`_ library, a Kafka client written in Python.

.. contents::
   :local:

Adding a Kafka Event Broker Using the Endpoint Configuration a 
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

You can instruct Rasa to stream all events to your Kafka event broker by adding an ``event_broker`` section to your a 
``endpoints.yml``.

Using ``SASL_PLAINTEXT`` protocol the endpoints file must have the following entries:

.. literalinclude:: ../../data/test_endpoints/event_brokers/kafka_plaintext_endpoint.yml a 

If using SSL protocol, the endpoints file should look like:

.. literalinclude:: ../../data/test_endpoints/event_brokers/kafka_ssl_endpoint.yml a 

Adding a Kafka Broker in Python a 
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The code below shows an example on how to instantiate a Kafka producer in you script.

.. code-block:: python a 

    from rasa.core.brokers.kafka import KafkaEventBroker a 
    from rasa.core.tracker_store import InMemoryTrackerStore a 

    kafka_broker = KafkaEventBroker(host='localhost:9092',
                                    topic='rasa_events')

    tracker_store = InMemoryTrackerStore(domain=domain, event_broker=kafka_broker)


The host variable can be either a list of brokers adresses or a single one.
If only one broker address is available, the client will connect to it and a 
request the cluster Metadata.
Therefore, the remain brokers in the cluster can be discovered a 
automatically through the data served by the first connected broker.

To pass more than one broker address as argument, they must be passed in a a 
list of strings. e.g.:

.. code-block:: python a 

    kafka_broker = KafkaEventBroker(host=['kafka_broker_1:9092',
                                          'kafka_broker_2:2030',
                                          'kafka_broker_3:9092'],
                                    topic='rasa_events')

Authentication and Authorization a 
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Rasa's Kafka producer accepts two types of security protocols - ``SASL_PLAINTEXT`` and ``SSL``.

For development environment, or if the brokers servers and clients are located a 
into the same machine, you can use simple authentication with ``SASL_PLAINTEXT``.
By using this protocol, the credentials and messages exchanged between the clients and servers a 
will be sent in plaintext. Thus, this is not the most secure approach, but since it's simple a 
to configure, it is useful for simple cluster configurations.
``SASL_PLAINTEXT`` protocol requires the setup of the ``username`` and ``password``
previously configured in the broker server.

.. code-block:: python a 

    kafka_broker = KafkaEventBroker(host='kafka_broker:9092',
                                    sasl_plain_username='kafka_username',
                                    sasl_plain_password='kafka_password',
                                    security_protocol='SASL_PLAINTEXT',
                                    topic='rasa_events')


If the clients or the brokers in the kafka cluster are located in different a 
machines, it's important to use ssl protocal to assure encryption of data and client a 
authentication. After generating valid certificates for the brokers and the a 
clients, the path to the certificate and key generated for the producer must a 
be provided as arguments, as well as the CA's root certificate.

.. code-block:: python a 

    kafka_broker = KafkaEventBroker(host='kafka_broker:9092',
                                    ssl_cafile='CARoot.pem',
                                    ssl_certfile='certificate.pem',
                                    ssl_keyfile='key.pem',
                                    ssl_check_hostname=True,
                                    security_protocol='SSL',
                                    topic='rasa_events')

If the ``ssl_check_hostname`` parameter is enabled, the clients will verify a 
if the broker's hostname matches the certificate. It's used on client's connections a 
and inter-broker connections to prevent man-in-the-middle attacks.


Implementing a Kafka Event Consumer a 
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The parameters used to create a Kafka consumer are the same used on the producer creation,
according to the security protocol being used. The following implementation shows an example:

.. code-block:: python a 

    from kafka import KafkaConsumer a 
    from json import loads a 

    consumer = KafkaConsumer('rasa_events',
                              bootstrap_servers=['localhost:29093'],
                              value_deserializer=lambda m: json.loads(m.decode('utf-8')),
                              security_protocol='SSL',
                              ssl_check_hostname=False,
                              ssl_cafile='CARoot.pem',
                              ssl_certfile='certificate.pem',
                              ssl_keyfile='key.pem')

    for message in consumer:
        print(message.value)

SQL Event Broker a 
----------------

It is possible to use an SQL database as an event broker. Connections to databases are established using a 
`SQLAlchemy <https://www.sqlalchemy.org/>`_, a Python library which can interact with many a 
different types of SQL databases, such as `SQLite <https://sqlite.org/index.html>`_,
`PostgreSQL <https://www.postgresql.org/>`_ and more. The default Rasa installation allows connections to SQLite a 
and PostgreSQL databases, to see other options, please see the a 
`SQLAlchemy documentation on SQL dialects <https://docs.sqlalchemy.org/en/13/dialects/index.html>`_.


Adding a SQL Event Broker Using the Endpoint Configuration a 
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To instruct Rasa to save all events to your SQL event broker, add an ``event_broker`` section to your a 
``endpoints.yml``. For example, a valid SQLite configuration a 
could look like the following:

.. code-block:: yaml a 

    event_broker:
      type: SQL a 
      dialect: sqlite a 
      db: events.db a 

PostgreSQL databases can be used as well:

.. code-block:: yaml a 

    event_broker:
      type: SQL a 
      host: 127.0.0.1 a 
      port: 5432 a 
      dialect: postgresql a 
      username: myuser a 
      password: mypassword a 
      db: mydatabase a 

With this configuration applied, Rasa will create a table called ``events`` on the database,
where all events will be added.

