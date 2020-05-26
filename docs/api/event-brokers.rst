:desc: Find out how open source chatbot framework Rasa allows a
       you to stream events to a message broker. a
 a
.. _event-brokers: a
 a
Event Brokers a
============= a
 a
.. edit-link:: a
 a
An event broker allows you to connect your running assistant to other services that process the data coming  a
in from conversations. For example, you could `connect your live assistant to  a
Rasa X <https://rasa.com/docs/rasa-x/installation-and-setup/existing-deployment/>`_ a
to review and annotate conversations or forward messages to an external analytics a
service. The event broker publishes messages to a message streaming service,  a
also known as a message broker, to forward Rasa :ref:`events` from the Rasa server to other services. a
 a
.. contents:: a
   :local: a
   :depth: 1 a
 a
Format a
------ a
 a
All events are streamed to the broker as serialised dictionaries every time a
the tracker updates its state. An example event emitted from the ``default`` a
tracker looks like this: a
 a
.. code-block:: json a
 a
    { a
        "sender_id": "default", a
        "timestamp": 1528402837.617099, a
        "event": "bot", a
        "text": "what your bot said", a
        "data": "some data about e.g. attachments" a
        "metadata" { a
              "a key": "a value", a
         } a
    } a
 a
The ``event`` field takes the event's ``type_name`` (for more on event a
types, check out the :ref:`events` docs). a
 a
 a
.. _event-brokers-pika: a
 a
Pika Event Broker a
----------------- a
 a
The example implementation we're going to show you here uses a
`Pika <https://pika.readthedocs.io>`_ , the Python client library for a
`RabbitMQ <https://www.rabbitmq.com>`_. a
 a
.. contents:: a
   :local: a
 a
Adding a Pika Event Broker Using the Endpoint Configuration a
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ a
 a
You can instruct Rasa to stream all events to your Pika event broker by adding an ``event_broker`` section to your a
``endpoints.yml``: a
 a
.. literalinclude:: ../../data/test_endpoints/event_brokers/pika_endpoint.yml a
 a
Rasa will automatically start streaming events when you restart the Rasa server. a
 a
 a
Adding a Pika Event Broker in Python a
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ a
 a
Here is how you add it using Python code: a
 a
.. code-block:: python a
 a
    from rasa.core.brokers.pika import PikaEventBroker a
    from rasa.core.tracker_store import InMemoryTrackerStore a
 a
    pika_broker = PikaEventBroker('localhost', a
                                  'username', a
                                  'password', a
                                  queues=['rasa_events']) a
 a
    tracker_store = InMemoryTrackerStore(domain=domain, event_broker=pika_broker) a
 a
 a
Implementing a Pika Event Consumer a
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ a
 a
You need to have a RabbitMQ server running, as well as another application a
that consumes the events. This consumer to needs to implement Pika's a
``start_consuming()`` method with a ``callback`` action. Here's a simple a
example: a
 a
.. code-block:: python a
 a
    import json a
    import pika a
 a
 a
    def _callback(self, ch, method, properties, body): a
            # Do something useful with your incoming message body here, e.g. a
            # saving it to a database a
            print('Received event {}'.format(json.loads(body))) a
 a
    if __name__ == '__main__': a
 a
        # RabbitMQ credentials with username and password a
        credentials = pika.PlainCredentials('username', 'password') a
 a
        # Pika connection to the RabbitMQ host - typically 'rabbit' in a a
        # docker environment, or 'localhost' in a local environment a
        connection = pika.BlockingConnection( a
            pika.ConnectionParameters('rabbit', credentials=credentials)) a
 a
        # start consumption of channel a
        channel = connection.channel() a
        channel.basic_consume(_callback, a
                              queue='rasa_events', a
                              no_ack=True) a
        channel.start_consuming() a
 a
Kafka Event Broker a
------------------ a
 a
It is possible to use `Kafka <https://kafka.apache.org/>`_ as main broker for your a
events. In this example we are going to use the `python-kafka <https://kafka-python a
.readthedocs.io/en/master/usage.html>`_ library, a Kafka client written in Python. a
 a
.. contents:: a
   :local: a
 a
Adding a Kafka Event Broker Using the Endpoint Configuration a
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ a
 a
You can instruct Rasa to stream all events to your Kafka event broker by adding an ``event_broker`` section to your a
``endpoints.yml``. a
 a
Using ``SASL_PLAINTEXT`` protocol the endpoints file must have the following entries: a
 a
.. literalinclude:: ../../data/test_endpoints/event_brokers/kafka_plaintext_endpoint.yml a
 a
If using SSL protocol, the endpoints file should look like: a
 a
.. literalinclude:: ../../data/test_endpoints/event_brokers/kafka_ssl_endpoint.yml a
 a
Adding a Kafka Broker in Python a
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ a
 a
The code below shows an example on how to instantiate a Kafka producer in you script. a
 a
.. code-block:: python a
 a
    from rasa.core.brokers.kafka import KafkaEventBroker a
    from rasa.core.tracker_store import InMemoryTrackerStore a
 a
    kafka_broker = KafkaEventBroker(host='localhost:9092', a
                                    topic='rasa_events') a
 a
    tracker_store = InMemoryTrackerStore(domain=domain, event_broker=kafka_broker) a
 a
 a
The host variable can be either a list of brokers adresses or a single one. a
If only one broker address is available, the client will connect to it and a
request the cluster Metadata. a
Therefore, the remain brokers in the cluster can be discovered a
automatically through the data served by the first connected broker. a
 a
To pass more than one broker address as argument, they must be passed in a a
list of strings. e.g.: a
 a
.. code-block:: python a
 a
    kafka_broker = KafkaEventBroker(host=['kafka_broker_1:9092', a
                                          'kafka_broker_2:2030', a
                                          'kafka_broker_3:9092'], a
                                    topic='rasa_events') a
 a
Authentication and Authorization a
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ a
 a
Rasa's Kafka producer accepts two types of security protocols - ``SASL_PLAINTEXT`` and ``SSL``. a
 a
For development environment, or if the brokers servers and clients are located a
into the same machine, you can use simple authentication with ``SASL_PLAINTEXT``. a
By using this protocol, the credentials and messages exchanged between the clients and servers a
will be sent in plaintext. Thus, this is not the most secure approach, but since it's simple a
to configure, it is useful for simple cluster configurations. a
``SASL_PLAINTEXT`` protocol requires the setup of the ``username`` and ``password`` a
previously configured in the broker server. a
 a
.. code-block:: python a
 a
    kafka_broker = KafkaEventBroker(host='kafka_broker:9092', a
                                    sasl_plain_username='kafka_username', a
                                    sasl_plain_password='kafka_password', a
                                    security_protocol='SASL_PLAINTEXT', a
                                    topic='rasa_events') a
 a
 a
If the clients or the brokers in the kafka cluster are located in different a
machines, it's important to use ssl protocal to assure encryption of data and client a
authentication. After generating valid certificates for the brokers and the a
clients, the path to the certificate and key generated for the producer must a
be provided as arguments, as well as the CA's root certificate. a
 a
.. code-block:: python a
 a
    kafka_broker = KafkaEventBroker(host='kafka_broker:9092', a
                                    ssl_cafile='CARoot.pem', a
                                    ssl_certfile='certificate.pem', a
                                    ssl_keyfile='key.pem', a
                                    ssl_check_hostname=True, a
                                    security_protocol='SSL', a
                                    topic='rasa_events') a
 a
If the ``ssl_check_hostname`` parameter is enabled, the clients will verify a
if the broker's hostname matches the certificate. It's used on client's connections a
and inter-broker connections to prevent man-in-the-middle attacks. a
 a
 a
Implementing a Kafka Event Consumer a
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ a
 a
The parameters used to create a Kafka consumer are the same used on the producer creation, a
according to the security protocol being used. The following implementation shows an example: a
 a
.. code-block:: python a
 a
    from kafka import KafkaConsumer a
    from json import loads a
 a
    consumer = KafkaConsumer('rasa_events', a
                              bootstrap_servers=['localhost:29093'], a
                              value_deserializer=lambda m: json.loads(m.decode('utf-8')), a
                              security_protocol='SSL', a
                              ssl_check_hostname=False, a
                              ssl_cafile='CARoot.pem', a
                              ssl_certfile='certificate.pem', a
                              ssl_keyfile='key.pem') a
 a
    for message in consumer: a
        print(message.value) a
 a
SQL Event Broker a
---------------- a
 a
It is possible to use an SQL database as an event broker. Connections to databases are established using a
`SQLAlchemy <https://www.sqlalchemy.org/>`_, a Python library which can interact with many a
different types of SQL databases, such as `SQLite <https://sqlite.org/index.html>`_, a
`PostgreSQL <https://www.postgresql.org/>`_ and more. The default Rasa installation allows connections to SQLite a
and PostgreSQL databases, to see other options, please see the a
`SQLAlchemy documentation on SQL dialects <https://docs.sqlalchemy.org/en/13/dialects/index.html>`_. a
 a
 a
Adding a SQL Event Broker Using the Endpoint Configuration a
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ a
 a
To instruct Rasa to save all events to your SQL event broker, add an ``event_broker`` section to your a
``endpoints.yml``. For example, a valid SQLite configuration a
could look like the following: a
 a
.. code-block:: yaml a
 a
    event_broker: a
      type: SQL a
      dialect: sqlite a
      db: events.db a
 a
PostgreSQL databases can be used as well: a
 a
.. code-block:: yaml a
 a
    event_broker: a
      type: SQL a
      url: 127.0.0.1 a
      port: 5432 a
      dialect: postgresql a
      username: myuser a
      password: mypassword a
      db: mydatabase a
 a
With this configuration applied, Rasa will create a table called ``events`` on the database, a
where all events will be added. a
 a