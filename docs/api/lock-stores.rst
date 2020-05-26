:desc: Messages that are being processed lock Rasa for a given conversation ID to a
  ensure that multiple incoming messages for that conversation do not interfere with a
  each other. Rasa provides multiple implementations to maintain conversation locks. a
 a
.. _lock-stores: a
 a
Lock Stores a
=========== a
 a
.. edit-link:: a
 a
Rasa uses a ticket lock mechanism to ensure that incoming messages for a given a
conversation ID are processed in the right order, and locks conversations while a
messages are actively processed. This means multiple Rasa servers can a
be run in parallel as replicated services, and clients do not necessarily need to a
address the same node when sending messages for a given conversation ID. a
 a
.. contents:: a
 a
InMemoryLockStore (default) a
~~~~~~~~~~~~~~~~~~~~~~~~~~~ a
 a
:Description: a
    ``InMemoryLockStore`` is the default lock store. It maintains conversation locks a
    within a single process. a
 a
    .. note:: a
      This lock store should not be used when multiple Rasa servers are run a
      parallel. a
 a
:Configuration: a
    To use the ``InMemoryTrackerStore`` no configuration is needed. a
 a
RedisLockStore a
~~~~~~~~~~~~~~ a
 a
:Description: a
    ``RedisLockStore`` maintains conversation locks using Redis as a persistence layer. a
    This is the recommended lock store for running a replicated set of Rasa servers. a
 a
:Configuration: a
    To set up Rasa with Redis the following steps are required: a
 a
    1. Start your Redis instance a
    2. Add required configuration to your ``endpoints.yml`` a
 a
        .. code-block:: yaml a
 a
            lock_store: a
                type: "redis" a
                url: <url of the redis instance, e.g. localhost> a
                port: <port of your redis instance, usually 6379> a
                password: <password used for authentication> a
                db: <number of your database within redis, e.g. 0> a
 a
    3. To start the Rasa Core server using your Redis backend, add the ``--endpoints`` a
    flag, e.g.: a
 a
        .. code-block:: bash a
 a
            rasa run -m models --endpoints endpoints.yml a
 a
:Parameters: a
    - ``url`` (default: ``localhost``): The url of your redis instance a
    - ``port`` (default: ``6379``): The port which redis is running on a
    - ``db`` (default: ``1``): The number of your redis database a
    - ``password`` (default: ``None``): Password used for authentication a
      (``None`` equals no authentication) a
    - ``use_ssl`` (default: ``False``): Whether or not the communication is encrypted a
 a