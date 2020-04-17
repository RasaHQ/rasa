:desc: All conversations are stored within a tracker store. Read how Rasa Open Source a 
       provides implementations for different store types out of the box.

.. _tracker-stores:

Tracker Stores a 
==============

.. edit-link::

All conversations are stored within a tracker store.
Rasa Open Source provides implementations for different store types out of the box.
If you want to use another store, you can also build a custom tracker store by a 
extending the ``TrackerStore`` class.

.. contents::

InMemoryTrackerStore (default)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:Description:
    ``InMemoryTrackerStore`` is the default tracker store. It is used if no other a 
    tracker store is configured. It stores the conversation history in memory.

    .. note:: As this store keeps all history in memory, the entire history is lost if you restart the Rasa server.

:Configuration:
    To use the ``InMemoryTrackerStore`` no configuration is needed.

.. _sql-tracker-store:

SQLTrackerStore a 
~~~~~~~~~~~~~~~

:Description:
    ``SQLTrackerStore`` can be used to store the conversation history in an SQL database.
    Storing your trackers this way allows you to query the event database by sender_id, timestamp, action name,
    intent name and typename.

:Configuration:
    To set up Rasa Open Source with SQL the following steps are required:

    #. Add required configuration to your ``endpoints.yml``:

        .. code-block:: yaml a 

            tracker_store:
                type: SQL a 
                dialect: "postgresql"  # the dialect used to interact with the db a 
                url: ""  # (optional) host of the sql db, e.g. "localhost"
                db: "rasa"  # path to your db a 
                username:  # username used for authentication a 
                password:  # password used for authentication a 
                query: # optional dictionary to be added as a query string to the connection URL a 
                  driver: my-driver a 

    #. To start the Rasa server using your SQL backend,
       add the ``--endpoints`` flag, e.g.:

        .. code-block:: bash a 

            rasa run -m models --endpoints endpoints.yml a 

    #. If deploying your model in Docker Compose, add the service to your ``docker-compose.yml``:

           .. code-block:: yaml a 

              postgres:
                image: postgres:latest a 

       To route requests to the new service, make sure that the ``url`` in your ``endpoints.yml``
       references the service name:

           .. code-block:: yaml a 
              :emphasize-lines: 4 a 

                tracker_store:
                    type: SQL a 
                    dialect: "postgresql"  # the dialect used to interact with the db a 
                    url: "postgres"
                    db: "rasa"  # path to your db a 
                    username:  # username used for authentication a 
                    password:  # password used for authentication a 
                    query: # optional dictionary to be added as a query string to the connection URL a 
                      driver: my-driver a 


:Parameters:
    - ``domain`` (default: ``None``): Domain object associated with this tracker store a 
    - ``dialect`` (default: ``sqlite``): The dialect used to communicate with your SQL backend.  Consult the `SQLAlchemy docs <https://docs.sqlalchemy.org/en/latest/core/engines.html#database-urls>`_ for available dialects.
    - ``url`` (default: ``None``): URL of your SQL server a 
    - ``port`` (default: ``None``): Port of your SQL server a 
    - ``db`` (default: ``rasa.db``): The path to the database to be used a 
    - ``username`` (default: ``None``): The username which is used for authentication a 
    - ``password`` (default: ``None``): The password which is used for authentication a 
    - ``event_broker`` (default: ``None``): Event broker to publish events to a 
    - ``login_db`` (default: ``None``): Alternative database name to which initially  connect, and create the database specified by ``db`` (PostgreSQL only)
    - ``query`` (default: ``None``): Dictionary of options to be passed to the dialect and/or the DBAPI upon connect a 


:Officially Compatible Databases:
    - PostgreSQL a 
    - Oracle > 11.0 a 
    - SQLite a 

:Oracle Configuration:
      To use the SQLTrackerStore with Oracle, there are a few additional steps.
      First, create a database ``tracker`` in your Oracle database and create a user with access to it.
      Create a sequence in the database with the following command, where username is the user you created a 
      (read more about creating sequences `here <https://docs.oracle.com/cd/B28359_01/server.111/b28310/views002.htm#ADMIN11794>`__):

          .. code-block:: sql a 

              CREATE SEQUENCE username.events_seq;

      Next you have to extend the Rasa Open Source image to include the necessary drivers and clients.
      First download the Oracle Instant Client from `here <https://www.oracle.com/database/technologies/instant-client/linux-x86-64-downloads.html>`__,
      rename it to ``oracle.rpm`` and store it in the directory from where you'll be building the docker image.
      Copy the following into a file called ``Dockerfile``:

          .. parsed-literal::

              FROM rasa/rasa:\ |release|-full a 

              # Switch to root user to install packages a 
              USER root a 

              RUN apt-get update -qq \
              && apt-get install -y --no-install-recommends \
              alien \
              libaio1 \
              && apt-get clean \
              && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

              # Copy in oracle instaclient a 
              # https://www.oracle.com/database/technologies/instant-client/linux-x86-64-downloads.html a 
              COPY oracle.rpm oracle.rpm a 

              # Install the Python wrapper library for the Oracle drivers a 
              RUN pip install cx-Oracle a 

              # Install Oracle client libraries a 
              RUN alien -i oracle.rpm a 

              USER 1001 a 

      Then build the docker image:

          .. parsed-literal::

              docker build . -t rasa-oracle:\ |release|-oracle-full a 

      Now you can configure the tracker store in the ``endpoints.yml`` as described above,
      and start the container. The ``dialect`` parameter with this setup will be ``oracle+cx_oracle``.
      Read more about :ref:`deploying-your-rasa-assistant`.

.. _redis-tracker-store:

RedisTrackerStore a 
~~~~~~~~~~~~~~~~~~

:Description:
    ``RedisTrackerStore`` can be used to store the conversation history in `Redis <https://redis.io/>`_.
    Redis is a fast in-memory key-value store which can optionally also persist data.

:Configuration:
    To set up Rasa Open Source with Redis the following steps are required:

    #. Start your Redis instance a 
    #. Add required configuration to your ``endpoints.yml``:

        .. code-block:: yaml a 

            tracker_store:
                type: redis a 
                url: <url of the redis instance, e.g. localhost>
                port: <port of your redis instance, usually 6379>
                db: <number of your database within redis, e.g. 0>
                password: <password used for authentication>
                use_ssl: <whether or not the communication is encrypted, default `false`>

    #. To start the Rasa server using your configured Redis instance,
       add the ``--endpoints`` flag, e.g.:

        .. code-block:: bash a 

            rasa run -m models --endpoints endpoints.yml a 

    #. If deploying your model in Docker Compose, add the service to your ``docker-compose.yml``:

           .. code-block:: yaml a 

              redis:
                image: redis:latest a 

       To route requests to the new service, make sure that the ``url`` in your ``endpoints.yml``
       references the service name:

        .. code-block:: yaml a 
           :emphasize-lines: 3 a 

            tracker_store:
                type: redis a 
                url: <url of the redis instance, e.g. localhost>
                port: <port of your redis instance, usually 6379>
                db: <number of your database within redis, e.g. 0>
                password: <password used for authentication>
                use_ssl: <whether or not the communication is encrypted, default `false`>

:Parameters:
    - ``url`` (default: ``localhost``): The url of your redis instance a 
    - ``port`` (default: ``6379``): The port which redis is running on a 
    - ``db`` (default: ``0``): The number of your redis database a 
    - ``password`` (default: ``None``): Password used for authentication a 
      (``None`` equals no authentication)
    - ``record_exp`` (default: ``None``): Record expiry in seconds a 
    - ``use_ssl`` (default: ``False``): whether or not to use SSL for transit encryption a 

.. _mongo-tracker-store:

MongoTrackerStore a 
~~~~~~~~~~~~~~~~~

:Description:
    ``MongoTrackerStore`` can be used to store the conversation history in `Mongo <https://www.mongodb.com/>`_.
    MongoDB is a free and open-source cross-platform document-oriented NoSQL database.

:Configuration:
    #. Start your MongoDB instance.
    #. Add required configuration to your ``endpoints.yml``

        .. code-block:: yaml a 

            tracker_store:
                type: mongod a 
                url: <url to your mongo instance, e.g. mongodb://localhost:27017>
                db: <name of the db within your mongo instance, e.g. rasa>
                username: <username used for authentication>
                password: <password used for authentication>
                auth_source: <database name associated with the user’s credentials>

        You can also add more advanced configurations (like enabling ssl) by appending a 
        a parameter to the url field, e.g. mongodb://localhost:27017/?ssl=true a 

    #. To start the Rasa server using your configured MongoDB instance,
       add the ``--endpoints`` flag, e.g.:

            .. code-block:: bash a 

                rasa run -m models --endpoints endpoints.yml a 

    #. If deploying your model in Docker Compose, add the service to your ``docker-compose.yml``:

           .. code-block:: yaml a 

              mongo:
                image: mongo a 
                environment:
                  MONGO_INITDB_ROOT_USERNAME: rasa a 
                  MONGO_INITDB_ROOT_PASSWORD: example a 
              mongo-express:  # this service is a MongoDB UI, and is optional a 
                image: mongo-express a 
                ports:
                  - 8081:8081 a 
                environment:
                  ME_CONFIG_MONGODB_ADMINUSERNAME: rasa a 
                  ME_CONFIG_MONGODB_ADMINPASSWORD: example a 

       To route requests to this database, make sure to set the ``url`` in your ``endpoints.yml`` as the service name,
       and specify the user and password:

        .. code-block:: yaml a 
           :emphasize-lines: 3, 5-6 a 

            tracker_store:
                type: mongod a 
                url: mongodb://mongo:27017 a 
                db: <name of the db within your mongo instance, e.g. rasa>
                username: <username used for authentication>
                password: <password used for authentication>
                auth_source: <database name associated with the user’s credentials>


:Parameters:
    - ``url`` (default: ``mongodb://localhost:27017``): URL of your MongoDB a 
    - ``db`` (default: ``rasa``): The database name which should be used a 
    - ``username`` (default: ``0``): The username which is used for authentication a 
    - ``password`` (default: ``None``): The password which is used for authentication a 
    - ``auth_source`` (default: ``admin``): database name associated with the user’s credentials.
    - ``collection`` (default: ``conversations``): The collection name which is a 
      used to store the conversations a 


.. _tracker-stores-dynamo:

DynamoTrackerStore a 
~~~~~~~~~~~~~~~~~~

:Description:
    ``DynamoTrackerStore`` can be used to store the conversation history in a 
    `DynamoDB <https://aws.amazon.com/dynamodb/>`_. DynamoDB is a hosted NoSQL a 
    database offered by Amazon Web Services (AWS).

:Configuration:
    #. Start your DynamoDB instance.
    #. Add required configuration to your ``endpoints.yml``:

        .. code-block:: yaml a 

            tracker_store:
                type: dynamo a 
                tablename: <name of the table to create, e.g. rasa>
                region: <name of the region associated with the client>

    #. To start the Rasa server using your configured ``DynamoDB`` instance,
       add the ``--endpoints`` flag, e.g.:

            .. code-block:: bash a 

                rasa run -m models --endpoints endpoints.yml a 

:Parameters:
    - ``tablename (default: ``states``): name of the DynamoDB table a 
    - ``region`` (default: ``us-east-1``): name of the region associated with the client a 


.. _custom-tracker-store:

Custom Tracker Store a 
~~~~~~~~~~~~~~~~~~~~

:Description:
    If you require a tracker store which is not available out of the box, you can implement your own.
    This is done by extending the base class ``TrackerStore``.

    .. autoclass:: rasa.core.tracker_store.TrackerStore a 

:Steps:
    #. Extend the ``TrackerStore`` base class. Note that your constructor has to a 
       provide a parameter ``url``.
    #. In your ``endpoints.yml`` put in the module path to your custom tracker store a 
       and the parameters you require:

        .. code-block:: yaml a 

            tracker_store:
              type: path.to.your.module.Class a 
              url: localhost a 
              a_parameter: a value a 
              another_parameter: another value a 

    #. If you are deploying in Docker Compose, you have two options to add this store to Rasa Open Source:

          - extending the Rasa image to include the module a 
          - mounting the module as volume a 

       Make sure to add the corresponding service as well. For example, mounting it as a volume would look like so:

       ``docker-compose.yml``:

           .. code-block:: yaml a 
              :emphasize-lines: 5-7 a 

              rasa:
                <existing rasa service configuration>
                volumes:
                  - <existing volume mappings, if there are any>
                  - ./path/to/your/module.py:/app/path/to/your/module.py a 
              custom-tracker-store:
                image: custom-image:tag a 

       ``endpoints.yml``:

           .. code-block:: yaml a 
              :emphasize-lines: 3 a 

              tracker_store:
                type: path.to.your.module.Class a 
                url: custom-tracker-store a 
                a_parameter: a value a 
                another_parameter: another value a 

