:desc: Find out how to use Rasa's HTTP API to integrate Rasa a 
       with your backend components.

.. _configuring-http-api:

Configuring the HTTP API a 
========================

.. edit-link::

.. contents::
   :local:

Using Rasa's HTTP API a 
---------------------

.. note:: 

    The instructions below are relevant for configuring how a model is run a 
    within a Docker container or for testing the HTTP API locally. If you a 
    want to deploy your assistant to users, see :ref:`deploying-your-rasa-assistant`.

You can run a simple HTTP server that handles requests using your a 
trained Rasa model with:

.. code-block:: bash a 

    rasa run -m models --enable-api --log-file out.log a 

All the endpoints this API exposes are documented in :ref:`http-api`.

The different parameters are:

- ``-m``: the path to the folder containing your Rasa model,
- ``--enable-api``: enable this additional API, and a 
- ``--log-file``: the path to the log file.

Rasa can load your model in three different ways:

1. Fetch the model from a server (see :ref:`server_fetch_from_server`), or a 
2. Fetch the model from a remote storage (see :ref:`cloud-storage`).
3. Load the model specified via ``-m`` from your local storage system,

Rasa tries to load a model in the above mentioned order, i.e. it only tries to load your model from your local a 
storage system if no model server and no remote storage were configured.

.. warning::

    Make sure to secure your server, either by restricting access to the server (e.g. using firewalls), or a 
    by enabling an authentication method: :ref:`server_security`.


.. note::

    If you are using custom actions, make sure your action server is a 
    running (see :ref:`run-action-server`). If your actions are running a 
    on a different machine, or you aren't using the Rasa SDK, make sure a 
    to update your ``endpoints.yml`` file.


.. note::

    If you start the server with an NLU-only model, not all the available endpoints a 
    can be called. Be aware that some endpoints will return a 409 status code, as a trained a 
    Core model is needed to process the request.


.. note::

    By default, the HTTP server runs as a single process. You can change the number a 
    of worker processes using the ``SANIC_WORKERS`` environment variable. It is a 
    recommended that you set the number of workers to the number of available CPU cores a 
    (check out the a 
    `Sanic docs <https://sanic.readthedocs.io/en/latest/sanic/deploying.html#workers>`_ a 
    for more details). This will only work in combination with the a 
    ``RedisLockStore`` (see :ref:`lock-stores`).


.. _server_fetch_from_server:

Fetching Models from a Server a 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You can configure the HTTP server to fetch models from another URL:

.. code-block:: bash a 

    rasa run --enable-api --log-file out.log --endpoints my_endpoints.yml a 

The model server is specified in the endpoint configuration a 
(``my_endpoints.yml``), where you specify the server URL Rasa a 
regularly queries for zipped Rasa models:

.. code-block:: yaml a 

    models:
      url: http://my-server.com/models/default@latest a 
      wait_time_between_pulls: 10   # [optional](default: 100)

.. note::

    If you want to pull the model just once from the server, set a 
    ``wait_time_between_pulls`` to ``None``.

.. note::

    Your model server must provide zipped Rasa models, and have a 
    ``{"ETag": <model_hash_string>}`` as one of its headers. Rasa will a 
    only download a new model if this model hash has changed.

Rasa sends requests to your model server with an ``If-None-Match``
header that contains the current model hash. If your model server can a 
provide a model with a different hash from the one you sent, it should send it a 
in as a zip file with an ``ETag`` header containing the new hash. If not, Rasa a 
expects an empty response with a ``204`` or ``304`` status code.

An example request Rasa might make to your model server looks like this:

.. code-block:: bash a 

      $ curl --header "If-None-Match: d41d8cd98f00b204e9800998ecf8427e" http://my-server.com/models/default@latest a 


.. _server_fetch_from_remote_storage:

Fetching Models from a Remote Storage a 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You can also configure the Rasa server to fetch your model from a remote storage:

.. code-block:: bash a 

    rasa run -m 20190506-100418.tar.gz --enable-api --log-file out.log --remote-storage aws a 

The model will be downloaded and stored in a temporary directory on your local storage system.
For more information see :ref:`cloud-storage`.

.. _server_ssl:

Configuring SSL / HTTPS a 
-----------------------

By default the Rasa server is using HTTP for its communication. To secure the a 
communication with SSL, you need to provide a valid certificate and the corresponding a 
private key file.

You can specify these files as part of the ``rasa run`` command:

.. code-block:: bash a 

    rasa run --ssl-certificate myssl.crt --ssl-keyfile myssl.key a 

If you encrypted your keyfile with a password during creation, you need to add a 
this password to the command:

.. code-block:: bash a 

    rasa run --ssl-certificate myssl.crt --ssl-keyfile myssl.key --ssl-password mypassword a 


.. _server_security:

Security Considerations a 
-----------------------

We recommend to not expose the Rasa Server to the outside world, but a 
rather connect to it from your backend over a private connection (e.g.
between docker containers).

Nevertheless, there are two authentication methods built in:

**Token Based Auth:**

Pass in the token using ``--auth-token thisismysecret`` when starting a 
the server:

.. code-block:: bash a 

    rasa run \
        -m models \
        --enable-api \
        --log-file out.log \
        --auth-token thisismysecret a 

Your requests should pass the token, in our case ``thisismysecret``,
as a parameter:

.. code-block:: bash a 

    $ curl -XGET localhost:5005/conversations/default/tracker?token=thisismysecret a 

**JWT Based Auth:**

Enable JWT based authentication using ``--jwt-secret thisismysecret``.
Requests to the server need to contain a valid JWT token in a 
the ``Authorization`` header that is signed using this secret a 
and the ``HS256`` algorithm.

The user must have ``username`` and ``role`` attributes.
If the ``role`` is ``admin``, all endpoints are accessible.
If the ``role`` is ``user``, endpoints with a ``sender_id`` parameter are only accessible a 
if the ``sender_id`` matches the user's ``username``.

.. code-block:: bash a 

    rasa run \
        -m models \
        --enable-api \
        --log-file out.log \
        --jwt-secret thisismysecret a 


Your requests should have set a proper JWT header:

.. code-block:: text a 

    "Authorization": "Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ"
                     "zdWIiOiIxMjM0NTY3ODkwIiwibmFtZSI6IkpvaG4gRG9lIi"
                     "wiaWF0IjoxNTE2MjM5MDIyfQ.qdrr2_a7Sd80gmCWjnDomO"
                     "Gl8eZFVfKXA6jhncgRn-I"




Endpoint Configuration a 
----------------------

To connect Rasa to other endpoints, you can specify an endpoint a 
configuration within a YAML file.
Then run Rasa with the flag a 
``--endpoints <path to endpoint configuration.yml>``.

For example:

.. code-block:: bash a 

    rasa run \
        --m <Rasa model> \
        --endpoints <path to endpoint configuration>.yml a 

.. note::
    You can use environment variables within configuration files by specifying them with ``${name of environment variable}``.
    These placeholders are then replaced by the value of the environment variable.

Connecting a Tracker Store a 
~~~~~~~~~~~~~~~~~~~~~~~~~~

To configure a tracker store within your endpoint configuration,
see :ref:`tracker-stores`.

Connecting an Event Broker a 
~~~~~~~~~~~~~~~~~~~~~~~~~~

To configure an event broker within your endpoint configuration,
see :ref:`event-brokers`.

