:desc: Find out how to use the HTTP API of Rasa to integrate
       with your backend components.

.. _running-the-server:

Running the Server
==================

.. contents::
   :local:

Running the HTTP server
-----------------------

You can run a simple HTTP server that handles requests using your
trained Rasa model with:

.. code-block:: bash

    rasa run -m models --enable-api --log-file out.log

All the endpoints this API exposes are documented in :ref:`http-api`.

The different parameters are:

- ``-m``: the path to the folder containing your Rasa model,
- ``--enable-api``: enable this additional API, and
- ``--log-file``: the path to the log file.

Rasa can load your model in three different ways:

1. Fetch the model from a server (see :ref:`server_fetch_from_server`), or
2. Fetch the model from a remote storage (see :ref:`cloud-storage`).
3. Load the model specified via ``-m`` from your local storage system,

Rasa tries to load a model in the above mentioned order, i.e. it only tries to load your model from your local
storage system if no model server and no remote storage were configured.

.. warning::

    Make sure to secure your server, either by restricting access to the server (e.g. using firewalls), or
    by enabling an authentication method: :ref:`server_security`.


.. note::

    If you are using custom actions, make sure your action server is
    running (see :ref:`run-action-server`). If your actions are running
    on a different machine, or you aren't using the Rasa SDK, make sure
    to update your ``endpoints.yml`` file.


.. note::

    If you start the server with an NLU-only model, not all the available endpoints
    can be called. Be aware that some endpoints will return a 409 status code, as a trained
    Core model is needed to process the request.


.. _server_fetch_from_server:

Fetching Models from a Server
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You can configure the HTTP server to fetch models from another URL:

.. code-block:: bash

    rasa run --enable-api --log-file out.log --endpoints my_endpoints.yml

The model server is specified in the endpoint configuration
(``my_endpoints.yml``), where you specify the server URL Rasa
regularly queries for zipped Rasa models:

.. code-block:: yaml

    models:
      url: http://my-server.com/models/default@latest
      wait_time_between_pulls: 10   # [optional](default: 100)

.. note::

    If you want to pull the model just once from the server, set
    ``wait_time_between_pulls`` to ``None``.

.. note::

    Your model server must provide zipped Rasa models, and have
    ``{"ETag": <model_hash_string>}`` as one of its headers. Rasa will
    only download a new model if this model hash has changed.

Rasa sends requests to your model server with an ``If-None-Match``
header that contains the current model hash. If your model server can
provide a model with a different hash from the one you sent, it should send it
in as a zip file with an ``ETag`` header containing the new hash. If not, Rasa
expects an empty response with a ``204`` or ``304`` status code.

An example request Rasa might make to your model server looks like this:

.. code-block:: bash

      $ curl --header "If-None-Match: d41d8cd98f00b204e9800998ecf8427e" http://my-server.com/models/default@latest


.. _server_fetch_from_remote_storage:

Fetching Models from a Remote Storage
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You can also configure the Rasa server to fetch your model from a remote storage:

.. code-block:: bash

    rasa run -m 20190506-100418.tar.gz --enable-api --log-file out.log --remote-storage aws

The model will be downloaded and stored in a temporary directory on your local storage system.
For more information see :ref:`cloud-storage`.


.. _server_security:

Security Considerations
-----------------------

We recommend to not expose the Rasa Server to the outside world, but
rather connect to it from your backend over a private connection (e.g.
between docker containers).

Nevertheless, there are two authentication methods built in:

**Token Based Auth:**

Pass in the token using ``--auth-token thisismysecret`` when starting
the server:

.. code-block:: bash

    rasa run \
        -m models \
        --enable-api \
        --log-file out.log \
        --auth-token thisismysecret

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

    rasa run \
        -m models \
        --enable-api \
        --log-file out.log \
        --jwt-secret thisismysecret


Your requests should have set a proper JWT header:

.. code-block:: text

    "Authorization": "Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ"
                     "zdWIiOiIxMjM0NTY3ODkwIiwibmFtZSI6IkpvaG4gRG9lIi"
                     "wiaWF0IjoxNTE2MjM5MDIyfQ.qdrr2_a7Sd80gmCWjnDomO"
                     "Gl8eZFVfKXA6jhncgRn-I"




Endpoint Configuration
----------------------

To connect Rasa to other endpoints, you can specify an endpoint
configuration within a YAML file.
Then run Rasa with the flag
``--endpoints <path to endpoint configuration.yml>``.

For example:

.. code-block:: bash

    rasa run \
        --m <Rasa model> \
        --endpoints <path to endpoint configuration>.yml

.. note::
    You can use environment variables within configuration files by specifying them with ``${name of environment variable}``.
    These placeholders are then replaced by the value of the environment variable.

Connecting a Tracker Store
~~~~~~~~~~~~~~~~~~~~~~~~~~

To configure a tracker store within your endpoint configuration,
see :ref:`tracker-stores`.

Connecting an Event Broker
~~~~~~~~~~~~~~~~~~~~~~~~~~

To configure an event broker within your endpoint configuration,
see :ref:`event-brokers`.
