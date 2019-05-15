:desc: All conversations are stored within a tracker store. Read how open source
       library Rasa Core provides implementations for different store types out
       of the box.

.. _tracker_store:


Tracker Stores
==============

All conversations are stored within a `tracker store`.
Rasa Core provides implementations for different store types out of the box.
If you want to use another store, you can also build a custom tracker store by extending the `TrackerStore` class.

.. contents::

InMemoryTrackerStore (default)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:Description:
    `InMemoryTrackerStore` is the default tracker store. It is used if no other tracker store is configured.
    It stores the conversation history in memory.

    .. note:: As this store keeps all history in memory the entire history is lost if you restart Rasa Core.

:Configuration:
    To use the `InMemoryTrackerStore` no configuration is needed.

SQLTrackerStore
~~~~~~~~~~~~~~~

:Description:
    ``SQLTrackerStore`` can be used to store the conversation history in an SQL database.
    Storing your trackers this way allows you to query the event database by sender_id, timestamp, action name,
    intent name and typename

:Configuration:
    To set up Rasa Core with SQL the following steps are required:

    1. Add required configuration to your `endpoints.yml`

        .. code-block:: yaml

            tracker_store:
                type: SQL
                dialect: "sqlite"  # the dialect used to interact with the db
                url: "localhost"  # (optional) host of the sql db
                db: "rasa.db"  # path to your db
                username:  # username used for authentication
                password:  # password used for authentication

    3. To start the Rasa Core server using your SQL backend,
       add the ``--endpoints`` flag, e.g.:

        .. code-block:: bash

            rasa run -m models --endpoints endpoints.yml
:Parameters:
    - ``dialect`` (default: ``sqlite``): The dialect used to communicate with your SQL backend.  `Consult the SQLAlchemy docs for available dialects <https://docs.sqlalchemy.org/en/latest/core/engines.html#database-urls>`_
    - ``url`` (default: ``None``): URL of your SQL database
    - ``db`` (default: ``rasa.db``): The path to the database to be used
    - ``username`` (default: ``None``): The username which is used for authentication
    - ``password`` (default: ``None``): The password which is used for authentication
    - ``collection`` (default: ``conversations``): The collection name which is
      used to store the conversations

RedisTrackerStore
~~~~~~~~~~~~~~~~~~

:Description:
    `RedisTrackerStore` can be used to store the conversation history in `Redis <https://redis.io/>`_.
    Redis is a fast in-memory key-value store which can optionally also persist data.

:Configuration:
    To set up Rasa Core with Redis the following steps are required:

    1. Start your Redis instance
    2. Add required configuration to your `endpoints.yml`

        .. code-block:: yaml

            tracker_store:
                type: redis
                url: <url of the redis instance, e.g. localhost>
                port: <port of your redis instance, usually 6379>
                db: <number of your database within redis, e.g. 0>
                password: <password used for authentication>

    3. To start the Rasa Core server using your configured Redis instance,
       add the :code:`--endpoints` flag, e.g.:

        .. code-block:: bash

            rasa run -m models --endpoints endpoints.yml
:Parameters:
    - ``url`` (default: ``localhost``): The url of your redis instance
    - ``port`` (default: ``6379``): The port which redis is running on
    - ``db`` (default: ``0``): The number of your redis database
    - ``password`` (default: ``None``): Password used for authentication
      (``None`` equals no authentication)
    - ``record_exp`` (default: ``None``): Record expiry in seconds

MongoTrackerStore
~~~~~~~~~~~~~~~~~

:Description:
    `MongoTrackerStore` can be used to store the conversation history in `Mongo <https://www.mongodb.com/>`_.
    MongoDB is a free and open-source cross-platform document-oriented NoSQL database.

:Configuration:
    1. Start your MongoDB instance.
    2. Add required configuration to your `endpoints.yml`

        .. code-block:: yaml

            tracker_store:
                type: mongod
                url: <url to your mongo instance, e.g. mongodb://localhost:27017>
                db: <name of the db within your mongo instance, e.g. rasa>
                username: <username used for authentication>
                password: <password used for authentication>
                auth_source: <database name associated with the user’s credentials>

        You can also add more advanced configurations (like enabling ssl) by appending
        a parameter to the url field, e.g. mongodb://localhost:27017/?ssl=true

    3. To start the Rasa Core server using your configured MongoDB instance,
           add the :code:`--endpoints` flag, e.g.:

            .. code-block:: bash

                rasa run -m models --endpoints endpoints.yml
:Parameters:
    - ``url`` (default: ``mongodb://localhost:27017``): URL of your MongoDB
    - ``db`` (default: ``rasa``): The database name which should be used
    - ``username`` (default: ``0``): The username which is used for authentication
    - ``password`` (default: ``None``): The password which is used for authentication
    - ``collection`` (default: ``conversations``): The collection name which is
      used to store the conversations
    - ``auth_source`` (default: ``admin``): database name associated with the user’s credentials.

Custom Tracker Store
~~~~~~~~~~~~~~~~~~~~

:Description:
    If you require a tracker store which is not available out of the box, you can implement your own.
    This is done by extending the base class `TrackerStore`.

    .. autoclass:: rasa.core.tracker_store.TrackerStore

:Steps:
    1. Extend the `TrackerStore` base class. Note that your constructor has to
       provide a parameter ``url``.
    2. In your endpoints.yml put in the module path to your custom tracker store
       and the parameters you require:

        .. code-block:: yaml

            tracker_store:
              type: path.to.your.module.Class
              url: localhost
              a_parameter: a value
              another_parameter: another value
