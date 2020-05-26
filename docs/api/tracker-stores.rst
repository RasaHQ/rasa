:desc: All conversations are stored within a tracker store. Read how Rasa Open Source a
       provides implementations for different store types out of the box. a
 a
.. _tracker-stores: a
 a
Tracker Stores a
============== a
 a
.. edit-link:: a
 a
All conversations are stored within a tracker store. a
Rasa Open Source provides implementations for different store types out of the box. a
If you want to use another store, you can also build a custom tracker store by a
extending the ``TrackerStore`` class. a
 a
.. contents:: a
 a
InMemoryTrackerStore (default) a
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ a
 a
:Description: a
    ``InMemoryTrackerStore`` is the default tracker store. It is used if no other a
    tracker store is configured. It stores the conversation history in memory. a
 a
    .. note:: As this store keeps all history in memory, the entire history is lost if you restart the Rasa server. a
 a
:Configuration: a
    To use the ``InMemoryTrackerStore`` no configuration is needed. a
 a
.. _sql-tracker-store: a
 a
SQLTrackerStore a
~~~~~~~~~~~~~~~ a
 a
:Description: a
    ``SQLTrackerStore`` can be used to store the conversation history in an SQL database. a
    Storing your trackers this way allows you to query the event database by sender_id, timestamp, action name, a
    intent name and typename. a
 a
:Configuration: a
    To set up Rasa Open Source with SQL the following steps are required: a
 a
    #. Add required configuration to your ``endpoints.yml``: a
 a
        .. code-block:: yaml a
 a
            tracker_store: a
                type: SQL a
                dialect: "postgresql"  # the dialect used to interact with the db a
                url: ""  # (optional) host of the sql db, e.g. "localhost" a
                db: "rasa"  # path to your db a
                username:  # username used for authentication a
                password:  # password used for authentication a
                query: # optional dictionary to be added as a query string to the connection URL a
                  driver: my-driver a
 a
    #. To start the Rasa server using your SQL backend, a
       add the ``--endpoints`` flag, e.g.: a
 a
        .. code-block:: bash a
 a
            rasa run -m models --endpoints endpoints.yml a
 a
    #. If deploying your model in Docker Compose, add the service to your ``docker-compose.yml``: a
 a
           .. code-block:: yaml a
 a
              postgres: a
                image: postgres:latest a
 a
       To route requests to the new service, make sure that the ``url`` in your ``endpoints.yml`` a
       references the service name: a
 a
           .. code-block:: yaml a
              :emphasize-lines: 4 a
 a
                tracker_store: a
                    type: SQL a
                    dialect: "postgresql"  # the dialect used to interact with the db a
                    url: "postgres" a
                    db: "rasa"  # path to your db a
                    username:  # username used for authentication a
                    password:  # password used for authentication a
                    query: # optional dictionary to be added as a query string to the connection URL a
                      driver: my-driver a
 a
 a
:Parameters: a
    - ``domain`` (default: ``None``): Domain object associated with this tracker store a
    - ``dialect`` (default: ``sqlite``): The dialect used to communicate with your SQL backend.  Consult the `SQLAlchemy docs <https://docs.sqlalchemy.org/en/latest/core/engines.html#database-urls>`_ for available dialects. a
    - ``url`` (default: ``None``): URL of your SQL server a
    - ``port`` (default: ``None``): Port of your SQL server a
    - ``db`` (default: ``rasa.db``): The path to the database to be used a
    - ``username`` (default: ``None``): The username which is used for authentication a
    - ``password`` (default: ``None``): The password which is used for authentication a
    - ``event_broker`` (default: ``None``): Event broker to publish events to a
    - ``login_db`` (default: ``None``): Alternative database name to which initially  connect, and create the database specified by ``db`` (PostgreSQL only) a
    - ``query`` (default: ``None``): Dictionary of options to be passed to the dialect and/or the DBAPI upon connect a
 a
 a
:Officially Compatible Databases: a
    - PostgreSQL a
    - Oracle > 11.0 a
    - SQLite a
 a
:Oracle Configuration: a
      To use the SQLTrackerStore with Oracle, there are a few additional steps. a
      First, create a database ``tracker`` in your Oracle database and create a user with access to it. a
      Create a sequence in the database with the following command, where username is the user you created a
      (read more about creating sequences `here <https://docs.oracle.com/cd/B28359_01/server.111/b28310/views002.htm#ADMIN11794>`__): a
 a
          .. code-block:: sql a
 a
              CREATE SEQUENCE username.events_seq; a
 a
      Next you have to extend the Rasa Open Source image to include the necessary drivers and clients. a
      First download the Oracle Instant Client from `here <https://www.oracle.com/database/technologies/instant-client/linux-x86-64-downloads.html>`__, a
      rename it to ``oracle.rpm`` and store it in the directory from where you'll be building the docker image. a
      Copy the following into a file called ``Dockerfile``: a
 a
          .. parsed-literal:: a
 a
              FROM rasa/rasa:\ |release|-full a
 a
              # Switch to root user to install packages a
              USER root a
 a
              RUN apt-get update -qq \ a
              && apt-get install -y --no-install-recommends \ a
              alien \ a
              libaio1 \ a
              && apt-get clean \ a
              && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/* a
 a
              # Copy in oracle instaclient a
              # https://www.oracle.com/database/technologies/instant-client/linux-x86-64-downloads.html a
              COPY oracle.rpm oracle.rpm a
 a
              # Install the Python wrapper library for the Oracle drivers a
              RUN pip install cx-Oracle a
 a
              # Install Oracle client libraries a
              RUN alien -i oracle.rpm a
 a
              USER 1001 a
 a
      Then build the docker image: a
 a
          .. parsed-literal:: a
 a
              docker build . -t rasa-oracle:\ |release|-oracle-full a
 a
      Now you can configure the tracker store in the ``endpoints.yml`` as described above, a
      and start the container. The ``dialect`` parameter with this setup will be ``oracle+cx_oracle``. a
      Read more about :ref:`deploying-your-rasa-assistant`. a
 a
.. _redis-tracker-store: a
 a
RedisTrackerStore a
~~~~~~~~~~~~~~~~~~ a
 a
:Description: a
    ``RedisTrackerStore`` can be used to store the conversation history in `Redis <https://redis.io/>`_. a
    Redis is a fast in-memory key-value store which can optionally also persist data. a
 a
:Configuration: a
    To set up Rasa Open Source with Redis the following steps are required: a
 a
    #. Start your Redis instance a
    #. Add required configuration to your ``endpoints.yml``: a
 a
        .. code-block:: yaml a
 a
            tracker_store: a
                type: redis a
                url: <url of the redis instance, e.g. localhost> a
                port: <port of your redis instance, usually 6379> a
                db: <number of your database within redis, e.g. 0> a
                password: <password used for authentication> a
                use_ssl: <whether or not the communication is encrypted, default `false`> a
 a
    #. To start the Rasa server using your configured Redis instance, a
       add the ``--endpoints`` flag, e.g.: a
 a
        .. code-block:: bash a
 a
            rasa run -m models --endpoints endpoints.yml a
 a
    #. If deploying your model in Docker Compose, add the service to your ``docker-compose.yml``: a
 a
           .. code-block:: yaml a
 a
              redis: a
                image: redis:latest a
 a
       To route requests to the new service, make sure that the ``url`` in your ``endpoints.yml`` a
       references the service name: a
 a
        .. code-block:: yaml a
           :emphasize-lines: 3 a
 a
            tracker_store: a
                type: redis a
                url: <url of the redis instance, e.g. localhost> a
                port: <port of your redis instance, usually 6379> a
                db: <number of your database within redis, e.g. 0> a
                password: <password used for authentication> a
                use_ssl: <whether or not the communication is encrypted, default `false`> a
 a
:Parameters: a
    - ``url`` (default: ``localhost``): The url of your redis instance a
    - ``port`` (default: ``6379``): The port which redis is running on a
    - ``db`` (default: ``0``): The number of your redis database a
    - ``password`` (default: ``None``): Password used for authentication a
      (``None`` equals no authentication) a
    - ``record_exp`` (default: ``None``): Record expiry in seconds a
    - ``use_ssl`` (default: ``False``): whether or not to use SSL for transit encryption a
 a
.. _mongo-tracker-store: a
 a
MongoTrackerStore a
~~~~~~~~~~~~~~~~~ a
 a
:Description: a
    ``MongoTrackerStore`` can be used to store the conversation history in `Mongo <https://www.mongodb.com/>`_. a
    MongoDB is a free and open-source cross-platform document-oriented NoSQL database. a
 a
:Configuration: a
    #. Start your MongoDB instance. a
    #. Add required configuration to your ``endpoints.yml`` a
 a
        .. code-block:: yaml a
 a
            tracker_store: a
                type: mongod a
                url: <url to your mongo instance, e.g. mongodb://localhost:27017> a
                db: <name of the db within your mongo instance, e.g. rasa> a
                username: <username used for authentication> a
                password: <password used for authentication> a
                auth_source: <database name associated with the user’s credentials> a
 a
        You can also add more advanced configurations (like enabling ssl) by appending a
        a parameter to the url field, e.g. mongodb://localhost:27017/?ssl=true a
 a
    #. To start the Rasa server using your configured MongoDB instance, a
       add the ``--endpoints`` flag, e.g.: a
 a
            .. code-block:: bash a
 a
                rasa run -m models --endpoints endpoints.yml a
 a
    #. If deploying your model in Docker Compose, add the service to your ``docker-compose.yml``: a
 a
           .. code-block:: yaml a
 a
              mongo: a
                image: mongo a
                environment: a
                  MONGO_INITDB_ROOT_USERNAME: rasa a
                  MONGO_INITDB_ROOT_PASSWORD: example a
              mongo-express:  # this service is a MongoDB UI, and is optional a
                image: mongo-express a
                ports: a
                  - 8081:8081 a
                environment: a
                  ME_CONFIG_MONGODB_ADMINUSERNAME: rasa a
                  ME_CONFIG_MONGODB_ADMINPASSWORD: example a
 a
       To route requests to this database, make sure to set the ``url`` in your ``endpoints.yml`` as the service name, a
       and specify the user and password: a
 a
        .. code-block:: yaml a
           :emphasize-lines: 3, 5-6 a
 a
            tracker_store: a
                type: mongod a
                url: mongodb://mongo:27017 a
                db: <name of the db within your mongo instance, e.g. rasa> a
                username: <username used for authentication> a
                password: <password used for authentication> a
                auth_source: <database name associated with the user’s credentials> a
 a
 a
:Parameters: a
    - ``url`` (default: ``mongodb://localhost:27017``): URL of your MongoDB a
    - ``db`` (default: ``rasa``): The database name which should be used a
    - ``username`` (default: ``0``): The username which is used for authentication a
    - ``password`` (default: ``None``): The password which is used for authentication a
    - ``auth_source`` (default: ``admin``): database name associated with the user’s credentials. a
    - ``collection`` (default: ``conversations``): The collection name which is a
      used to store the conversations a
 a
 a
.. _tracker-stores-dynamo: a
 a
DynamoTrackerStore a
~~~~~~~~~~~~~~~~~~ a
 a
:Description: a
    ``DynamoTrackerStore`` can be used to store the conversation history in a
    `DynamoDB <https://aws.amazon.com/dynamodb/>`_. DynamoDB is a hosted NoSQL a
    database offered by Amazon Web Services (AWS). a
 a
:Configuration: a
    #. Start your DynamoDB instance. a
    #. Add required configuration to your ``endpoints.yml``: a
 a
        .. code-block:: yaml a
 a
            tracker_store: a
                type: dynamo a
                tablename: <name of the table to create, e.g. rasa> a
                region: <name of the region associated with the client> a
 a
    #. To start the Rasa server using your configured ``DynamoDB`` instance, a
       add the ``--endpoints`` flag, e.g.: a
 a
            .. code-block:: bash a
 a
                rasa run -m models --endpoints endpoints.yml a
 a
:Parameters: a
    - ``tablename (default: ``states``): name of the DynamoDB table a
    - ``region`` (default: ``us-east-1``): name of the region associated with the client a
 a
 a
.. _custom-tracker-store: a
 a
Custom Tracker Store a
~~~~~~~~~~~~~~~~~~~~ a
 a
:Description: a
    If you require a tracker store which is not available out of the box, you can implement your own. a
    This is done by extending the base class ``TrackerStore``. a
 a
    .. autoclass:: rasa.core.tracker_store.TrackerStore a
 a
:Steps: a
    #. Extend the ``TrackerStore`` base class. Note that your constructor has to a
       provide a parameter ``url``. a
    #. In your ``endpoints.yml`` put in the module path to your custom tracker store a
       and the parameters you require: a
 a
        .. code-block:: yaml a
 a
            tracker_store: a
              type: path.to.your.module.Class a
              url: localhost a
              a_parameter: a value a
              another_parameter: another value a
 a
    #. If you are deploying in Docker Compose, you have two options to add this store to Rasa Open Source: a
 a
          - extending the Rasa image to include the module a
          - mounting the module as volume a
 a
       Make sure to add the corresponding service as well. For example, mounting it as a volume would look like so: a
 a
       ``docker-compose.yml``: a
 a
           .. code-block:: yaml a
              :emphasize-lines: 5-7 a
 a
              rasa: a
                <existing rasa service configuration> a
                volumes: a
                  - <existing volume mappings, if there are any> a
                  - ./path/to/your/module.py:/app/path/to/your/module.py a
              custom-tracker-store: a
                image: custom-image:tag a
 a
       ``endpoints.yml``: a
 a
           .. code-block:: yaml a
              :emphasize-lines: 3 a
 a
              tracker_store: a
                type: path.to.your.module.Class a
                url: custom-tracker-store a
                a_parameter: a value a
                another_parameter: another value a
 a