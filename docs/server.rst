:desc: Find out how to use the HTTP API of Rasa to integrate
       with your backend components.

.. _section_http:

Running the Server
==================



.. contents::
   :local:

Running the HTTP server
-----------------------

You can run a simple http server that handles requests using your
models with:

.. code-block:: bash

    rasa run core \
        --enable-api \
        -m models \
        -o out.log

All the endpoints this API exposes are documented in :ref:`http-api` .

The different parameters are:

- ``--enable-api``, enables this additional API
- ``-m``, which is the path to the folder containing your Rasa model.
- ``-o``, which is the path to the log file.


.. warning::

    Make sure to secure your server, either by restricting access to the server (e.g. using firewalls) or
    by enabling one of the authentication methods: :ref:`server_security`.


.. note::

    If you are using custom actions - make sure your action server is 
    running (see :ref:`run-action-server`). If your actions are running
    on a different machine, or you aren't using the Rasa SDk, make sure
    to update your ``endpoints.yml`` file.

Events
------
Events allow you to modify the internal state of the dialogue. This information
will be used to predict the next action. E.g. you can set slots (to store
information about the user) or restart the conversation.

You can return multiple events as part of your query, e.g.:

.. code-block:: bash

    $ curl -XPOST http://localhost:5005/conversations/default/tracker/events -d \
        '{"event": "slot", "name": "cuisine", "value": "mexican"}'


You can find a list of all events and their json representation
at :ref:`events`. You need to send these json formats to the endpoint to
log the event.


.. _server_security:

Security Considerations
-----------------------

We recommend to not expose the Rasa Server to the outside world but
rather connect to it from your backend over a private connection (e.g.
between docker containers).

Nevertheless, there are two authentication methods built in:

**Token Based Auth:**

Pass in the token using ``--auth-token thisismysecret`` when starting
the server:

.. code-block:: bash

    $ rasa run core \
        --enable-api \
        --auth-token thisismysecret \
        -m models \
        -o out.log

Your requests should pass the token, in our case ``thisismysecret``,
as a parameter:

.. code-block:: bash

    $ curl -XGET localhost:5005/conversations/default/tracker?token=thisismysecret

**JWT Based Auth:**

Enable JWT based authentication using ``--jwt-secret thisismysecret``.
Requests to the server need to contain a valid JWT token in
the ``Authorization`` header that is signed using this secret
and the ``HS256`` algorithm.

The user must have ``username`` and ``role`` attributes.
If the ``role`` is ``admin``, all endpoints are accessible.
If the ``role`` is ``user``, endpoints with a ``sender_id`` parameter are only accessible
if the ``sender_id`` matches the user's ``username``.

.. code-block:: bash

    $ rasa run core \
        --enable-api \
        --jwt-secret thisismysecret \
        -m models \
        -o out.log

Your requests should have set a proper JWT header:

.. code-block:: text

    "Authorization": "Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ"
                     "zdWIiOiIxMjM0NTY3ODkwIiwibmFtZSI6IkpvaG4gRG9lIi"
                     "wiaWF0IjoxNTE2MjM5MDIyfQ.qdrr2_a7Sd80gmCWjnDomO"
                     "Gl8eZFVfKXA6jhncgRn-I"


Connecting a Tracker Store
--------------------------

To configure a tracker store within your endpoint configuration,
please see :ref:`tracker_store`.

Connecting an Event Broker
--------------------------

To configure an event broker within your endpoint configuration,
please see :ref:`brokers`.
