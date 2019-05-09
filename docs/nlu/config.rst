:desc: Read more on configuring open source library Rasa NLU to access machine
       learning based prediction of intents and entities as a server.

.. _section_configuration:

Server Configuration
====================


.. note::

    Before you can use the server, you should train a model!
    See :ref:`training_your_model`


.. note::

    In older versions of Rasa NLU, the server and models were configured with a single file.
    Now, the server only takes command line arguments (see :ref:`server_parameters`).
    The configuration file only refers to the model that you want to train,
    i.e. the pipeline and components.


Running the server
------------------

You can run a simple HTTP server that handles requests using your model with:

.. code-block:: bash

    $ rasa run -m models

The server will look for existing models under the folder defined by
the ``-m`` parameter. By default the latest trained model will be loaded.

.. _server_parameters:

Server Parameters
-----------------

There are a number of parameters you can pass when running the server.

.. code-block:: console

    $ rasa run

Here is a quick overview:

.. program-output:: rasa run --help


.. _section_auth:

Authentication
--------------
To protect your server, you can specify a token in your Rasa NLU configuration,
by passing the ``--token`` argument when starting the server,
or by setting the ``RASA_NLU_TOKEN`` environment variable.
If set, this token must be passed as a query parameter in all requests, e.g. :

.. code-block:: bash

    $ curl localhost:5000/status?token=12345

CORS
----

By default CORS (cross-origin resource sharing) calls are not allowed. If you want to call your Rasa NLU server from another domain (for example from a training web UI) then you can whitelist that domain by adding it to the config value ``cors_origin``.

.. include:: feedback.inc
