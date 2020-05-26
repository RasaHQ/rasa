:desc: Find out how to use Rasa's HTTP API to integrate Rasa a
       with your backend components. a
 a
.. _configuring-http-api: a
 a
Configuring the HTTP API a
======================== a
 a
.. edit-link:: a
 a
.. contents:: a
   :local: a
 a
Using Rasa's HTTP API a
--------------------- a
 a
.. note::  a
 a
    The instructions below are relevant for configuring how a model is run a
    within a Docker container or for testing the HTTP API locally. If you a
    want to deploy your assistant to users, see :ref:`deploying-your-rasa-assistant`. a
 a
You can run a simple HTTP server that handles requests using your a
trained Rasa model with: a
 a
.. code-block:: bash a
 a
    rasa run -m models --enable-api --log-file out.log a
 a
All the endpoints this API exposes are documented in :ref:`http-api`. a
 a
The different parameters are: a
 a
- ``-m``: the path to the folder containing your Rasa model, a
- ``--enable-api``: enable this additional API, and a
- ``--log-file``: the path to the log file. a
 a
Rasa can load your model in three different ways: a
 a
1. Fetch the model from a server (see :ref:`server_fetch_from_server`), or a
2. Fetch the model from a remote storage (see :ref:`cloud-storage`). a
3. Load the model specified via ``-m`` from your local storage system, a
 a
Rasa tries to load a model in the above mentioned order, i.e. it only tries to load your model from your local a
storage system if no model server and no remote storage were configured. a
 a
.. warning:: a
 a
    Make sure to secure your server, either by restricting access to the server (e.g. using firewalls), or a
    by enabling an authentication method: :ref:`server_security`. a
 a
 a
.. note:: a
 a
    If you are using custom actions, make sure your action server is a
    running (see :ref:`run-action-server`). If your actions are running a
    on a different machine, or you aren't using the Rasa SDK, make sure a
    to update your ``endpoints.yml`` file. a
 a
 a
.. note:: a
 a
    If you start the server with an NLU-only model, not all the available endpoints a
    can be called. Be aware that some endpoints will return a 409 status code, as a trained a
    Core model is needed to process the request. a
 a
 a
.. note:: a
 a
    By default, the HTTP server runs as a single process. You can change the number a
    of worker processes using the ``SANIC_WORKERS`` environment variable. It is a
    recommended that you set the number of workers to the number of available CPU cores a
    (check out the a
    `Sanic docs <https://sanic.readthedocs.io/en/latest/sanic/deploying.html#workers>`_ a
    for more details). This will only work in combination with the a
    ``RedisLockStore`` (see :ref:`lock-stores`). a
 a
 a
.. _server_fetch_from_server: a
 a
Fetching Models from a Server a
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ a
 a
You can configure the HTTP server to fetch models from another URL: a
 a
.. code-block:: bash a
 a
    rasa run --enable-api --log-file out.log --endpoints my_endpoints.yml a
 a
The model server is specified in the endpoint configuration a
(``my_endpoints.yml``), where you specify the server URL Rasa a
regularly queries for zipped Rasa models: a
 a
.. code-block:: yaml a
 a
    models: a
      url: http://my-server.com/models/default@latest a
      wait_time_between_pulls: 10   # [optional](default: 100) a
 a
.. note:: a
 a
    If you want to pull the model just once from the server, set a
    ``wait_time_between_pulls`` to ``None``. a
 a
.. note:: a
 a
    Your model server must provide zipped Rasa models, and have a
    ``{"ETag": <model_hash_string>}`` as one of its headers. Rasa will a
    only download a new model if this model hash has changed. a
 a
Rasa sends requests to your model server with an ``If-None-Match`` a
header that contains the current model hash. If your model server can a
provide a model with a different hash from the one you sent, it should send it a
in as a zip file with an ``ETag`` header containing the new hash. If not, Rasa a
expects an empty response with a ``204`` or ``304`` status code. a
 a
An example request Rasa might make to your model server looks like this: a
 a
.. code-block:: bash a
 a
      $ curl --header "If-None-Match: d41d8cd98f00b204e9800998ecf8427e" http://my-server.com/models/default@latest a
 a
 a
.. _server_fetch_from_remote_storage: a
 a
Fetching Models from a Remote Storage a
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ a
 a
You can also configure the Rasa server to fetch your model from a remote storage: a
 a
.. code-block:: bash a
 a
    rasa run -m 20190506-100418.tar.gz --enable-api --log-file out.log --remote-storage aws a
 a
The model will be downloaded and stored in a temporary directory on your local storage system. a
For more information see :ref:`cloud-storage`. a
 a
.. _server_ssl: a
 a
Configuring SSL / HTTPS a
----------------------- a
 a
By default the Rasa server is using HTTP for its communication. To secure the a
communication with SSL, you need to provide a valid certificate and the corresponding a
private key file. a
 a
You can specify these files as part of the ``rasa run`` command: a
 a
.. code-block:: bash a
 a
    rasa run --ssl-certificate myssl.crt --ssl-keyfile myssl.key a
 a
If you encrypted your keyfile with a password during creation, you need to add a
this password to the command: a
 a
.. code-block:: bash a
 a
    rasa run --ssl-certificate myssl.crt --ssl-keyfile myssl.key --ssl-password mypassword a
 a
 a
.. _server_security: a
 a
Security Considerations a
----------------------- a
 a
We recommend to not expose the Rasa Server to the outside world, but a
rather connect to it from your backend over a private connection (e.g. a
between docker containers). a
 a
Nevertheless, there are two authentication methods built in: a
 a
**Token Based Auth:** a
 a
Pass in the token using ``--auth-token thisismysecret`` when starting a
the server: a
 a
.. code-block:: bash a
 a
    rasa run \ a
        -m models \ a
        --enable-api \ a
        --log-file out.log \ a
        --auth-token thisismysecret a
 a
Your requests should pass the token, in our case ``thisismysecret``, a
as a parameter: a
 a
.. code-block:: bash a
 a
    $ curl -XGET localhost:5005/conversations/default/tracker?token=thisismysecret a
 a
**JWT Based Auth:** a
 a
Enable JWT based authentication using ``--jwt-secret thisismysecret``. a
Requests to the server need to contain a valid JWT token in a
the ``Authorization`` header that is signed using this secret a
and the ``HS256`` algorithm. a
 a
The user must have ``username`` and ``role`` attributes. a
If the ``role`` is ``admin``, all endpoints are accessible. a
If the ``role`` is ``user``, endpoints with a ``sender_id`` parameter are only accessible a
if the ``sender_id`` matches the user's ``username``. a
 a
.. code-block:: bash a
 a
    rasa run \ a
        -m models \ a
        --enable-api \ a
        --log-file out.log \ a
        --jwt-secret thisismysecret a
 a
 a
Your requests should have set a proper JWT header: a
 a
.. code-block:: text a
 a
    "Authorization": "Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ" a
                     "zdWIiOiIxMjM0NTY3ODkwIiwibmFtZSI6IkpvaG4gRG9lIi" a
                     "wiaWF0IjoxNTE2MjM5MDIyfQ.qdrr2_a7Sd80gmCWjnDomO" a
                     "Gl8eZFVfKXA6jhncgRn-I" a
 a
 a
 a
 a
Endpoint Configuration a
---------------------- a
 a
To connect Rasa to other endpoints, you can specify an endpoint a
configuration within a YAML file. a
Then run Rasa with the flag a
``--endpoints <path to endpoint configuration.yml>``. a
 a
For example: a
 a
.. code-block:: bash a
 a
    rasa run \ a
        --m <Rasa model> \ a
        --endpoints <path to endpoint configuration>.yml a
 a
.. note:: a
    You can use environment variables within configuration files by specifying them with ``${name of environment variable}``. a
    These placeholders are then replaced by the value of the environment variable. a
 a
Connecting a Tracker Store a
~~~~~~~~~~~~~~~~~~~~~~~~~~ a
 a
To configure a tracker store within your endpoint configuration, a
see :ref:`tracker-stores`. a
 a
Connecting an Event Broker a
~~~~~~~~~~~~~~~~~~~~~~~~~~ a
 a
To configure an event broker within your endpoint configuration, a
see :ref:`event-brokers`. a
 a