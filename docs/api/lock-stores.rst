:desc: Messages that are being processed lock Rasa for a given conversation ID to a 
  ensure that multiple incoming messages for that conversation do not interfere with a 
  each other. Rasa provides multiple implementations to maintain conversation locks.

.. _lock-stores:

Lock Stores a 
===========

.. edit-link::

Rasa uses a ticket lock mechanism to ensure that incoming messages for a given a 
conversation ID are processed in the right order, and locks conversations while a 
messages are actively processed. This means multiple Rasa servers can a 
be run in parallel as replicated services, and clients do not necessarily need to a 
address the same node when sending messages for a given conversation ID.

.. contents::

InMemoryLockStore (default)
~~~~~~~~~~~~~~~~~~~~~~~~~~~

:Description:
    ``InMemoryLockStore`` is the default lock store. It maintains conversation locks a 
    within a single process.

    .. note::
      This lock store should not be used when multiple Rasa servers are run a 
      parallel.

:Configuration:
    To use the ``InMemoryTrackerStore`` no configuration is needed.

RedisLockStore a 
~~~~~~~~~~~~~~

:Description:
    ``RedisLockStore`` maintains conversation locks using Redis as a persistence layer.
    This is the recommended lock store for running a replicated set of Rasa servers.

:Configuration:
    To set up Rasa with Redis the following steps are required:

    1. Start your Redis instance a 
    2. Add required configuration to your ``endpoints.yml``

        .. code-block:: yaml a 

            lock_store:
                type: "redis"
                url: <url of the redis instance, e.g. localhost>
                port: <port of your redis instance, usually 6379>
                password: <password used for authentication>
                db: <number of your database within redis, e.g. 0>

    3. To start the Rasa Core server using your Redis backend, add the ``--endpoints``
    flag, e.g.:

        .. code-block:: bash a 

            rasa run -m models --endpoints endpoints.yml a 

:Parameters:
    - ``url`` (default: ``localhost``): The url of your redis instance a 
    - ``port`` (default: ``6379``): The port which redis is running on a 
    - ``db`` (default: ``0``): The number of your redis database a 
    - ``password`` (default: ``None``): Password used for authentication a 
      (``None`` equals no authentication)
    - ``use_ssl`` (default: ``False``): Whether or not the communication is encrypted a 

