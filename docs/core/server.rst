:desc: Find out how to use the HTTP API of Rasa Core to integrate
       with your backend components.

.. _section_http:

HTTP API
========

.. warning::

    To protect your conversational data, make sure to secure the server.
    Either by restricting access to the server (e.g. using firewalls) or
    by enabling one of the authentication methods: :ref:`server_security`.

.. note::

    Before you can use the server, you need to define a domain, create training
    data, and train a model. You can then use the trained model!
    See :ref:`quickstart` for an introduction.

    If you are looking for documentation on how to run custom actions -
    head over to :ref:`customactions`.


The HTTP api exists to make it easy for python and non-python
projects to interact with Rasa Core. The API allows you to modify
the trackers.


.. contents::


Running the HTTP server
-----------------------

You can run a simple http server that handles requests using your
models with:

.. code-block:: bash

    $ rasa run \
        --enable-api \
        -m models \
        -o out.log

The different parameters are:

- ``--enable-api``, enables this additional API
- ``-m``, which is the path to the folder containing your Rasa Core model
  and Rasa NLU model.
- ``-o``, which is the path to the log file.

.. note::

  If you are using custom actions - make sure to pass in the endpoint
  configuration for your action server as well using
  ``--endpoints endpoints.yml``.

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

We recommend to not expose the Rasa Core server to the outside world but
rather connect to it from your backend over a private connection (e.g.
between docker containers).

Nevertheless, there are two authentication methods built in:

**Token Based Auth:**

Pass in the token using ``--auth-token thisismysecret`` when starting
the server:

.. code-block:: bash

    $ rasa run \
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

    $ rasa run \
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

Endpoint Configuration
----------------------

To connect Rasa Core to other endpoints, you can specify an endpoint
configuration within a `YAML <https://en.wikipedia.org/wiki/YAML>`_ file.
Then run Rasa Core with the flag
``--endpoints <path to endpoint configuration.yml``.

For example:

.. code-block:: bash

    rasa run \
        --m <core model> \
        --endpoints <path to endpoint configuration>.yml

.. note::
    You can use environment variables within configuration files by specifying them with ``${name of environment variable}``.
    These placeholders are then replaced by the value of the environment variable.


Fetching Models From a Server
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You can also configure the http server to fetch models from another URL:

.. code-block:: bash

    $ rasa run \
        --enable-api \
        -m models \
        --endpoints my_endpoints.yaml \
        -o out.log

The model server is specified in the endpoint configuration
(``my_endpoints.yaml``), where you specify the server URL Rasa Core
regularly queries for zipped Rasa Core models:

.. code-block:: yaml

    models:
      url: http://my-server.com/models/default_core@latest
      wait_time_between_pulls:  10   # [optional](default: 100)

.. note::

    If you want to pull the model just once from the server, set
    ``wait_time_between_pulls`` to ``None``.

.. note::

    Your model server must provide zipped Rasa Core models, and have
    ``{"ETag": <model_hash_string>}`` as one of its headers. Core will
    only download a new model if this model hash changed.

Rasa Core sends requests to your model server with an ``If-None-Match``
header that contains the current model hash. If your model server can
provide a model with a different hash from the one you sent, it should send it
in as a zip file with an ``ETag`` header containing the new hash. If not, Rasa
Core expects an empty response with a ``204`` or ``304`` status code.

An example request Rasa Core might make to your model server looks like this:

.. code-block:: bash

      $ curl --header "If-None-Match: d41d8cd98f00b204e9800998ecf8427e" http://my-server.com/models/default_core@latest

Connecting Rasa NLU
~~~~~~~~~~~~~~~~~~~

To connect Rasa Core with a NLU server, you have to add the connection details
into your endpoint configuration file:

.. code-block:: yaml

    nlu:
        url: "http://<your nlu host>:5000"
        token: <token>  # [optional]
        token_name: <name of the token> # [optional] (default: token)

Then run Rasa Core with the ``--endpoints <path_to_your_endpoint_config>.yml``.
For example:

.. code-block:: bash

    rasa run -m models --endpoints endpoints.yml

Connecting a Tracker Store
~~~~~~~~~~~~~~~~~~~~~~~~~~

To configure a tracker store within your endpoint configuration,
please see :ref:`tracker_store`.

Connecting an Event Broker
~~~~~~~~~~~~~~~~~~~~~~~~~~

To configure an event broker within your endpoint configuration,
please see :ref:`brokers`.


Endpoints
---------

Documentation of the server API as
:download:`OpenAPI Spec <_static/spec/server.yml>`.

.. apidoc::
   :path: ../_static/spec/server.yml

.. include:: feedback.inc
