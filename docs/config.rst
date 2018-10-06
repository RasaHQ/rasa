:desc: Customizing Your Rasa NLU Configuration
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

You can run a simple http server that handles requests using your projects with :

.. code-block:: bash

    $ python -m rasa_nlu.server --path projects

The server will look for existing projects under the folder defined by
the ``path`` parameter. By default a project will load the latest
trained model.

.. _section_http_config:

Serving Multiple Apps
---------------------

Depending on your choice of backend, Rasa NLU can use quite a lot of memory.
So if you are serving multiple models in production, you want to serve these
from the same process & avoid duplicating the memory load.

.. note::

    Although this saves the backend from loading the same set of word vectors twice,
    if you have projects in multiple languages your memory usage will still be high.


As stated previously, Rasa NLU naturally handles serving multiple apps.
By default the server will load all projects found
under the ``path`` directory passed at run time. 

Rasa NLU naturally handles serving multiple apps, by default the server will load all projects found
under the directory specified with ``--path`` option. unless you have provide ``--pre_load`` option 
to load a specific project. 

.. code-block:: console

    $ # This will load all projects under projects/ directory
    $ python -m rasa_nlu.server -c config.yaml --path projects/ 

.. code-block:: console

    $ # This will load only hotels project under projects/ directory
    $ python -m rasa_nlu.server -c config.yaml --pre_load hotels --path projects/ 


The file structure under ``path directory`` is as follows:

.. code-block:: text

    - <path>
     - <project_A>
      - <model_XXXXXX>
      - <model_XXXXXX>
       ...
     - <project_B>
      - <model_XXXXXX>
       ...
      ...

You can specify which project to use in your ``/parse`` requests:

.. code-block:: console

    $ curl 'localhost:5000/parse?q=hello&project=my_restaurant_search_bot'

or

.. code-block:: console

    $ curl -XPOST localhost:5000/parse -d '{"q":"I am looking for Chinese food", "project":"my_restaurant_search_bot"}'

You can also specify the model you want to use for a given project, the default used being the latest trained:

.. code-block:: console

    $ curl -XPOST localhost:5000/parse -d '{"q":"I am looking for Chinese food", "project":"my_restaurant_search_bot", "model":<model_XXXXXX>}'

If no project is found by the server under the ``path`` directory, a ``"default"`` one will be used, using a simple fallback model.

.. _server_parameters:

Server Parameters
-----------------

There are a number of parameters you can pass when running the server.

.. code-block:: console

    $ python -m rasa_nlu.server

Here is a quick overview:

.. program-output:: python -m rasa_nlu.server --help


.. _section_auth:

Authentication
--------------
To protect your server, you can specify a token in your Rasa NLU configuration,
by passing the ``--token`` argument when starting the server,
or by setting the ``RASA_TOKEN`` environment variable.
If set, this token must be passed as a query parameter in all requests, e.g. :

.. code-block:: bash

    $ curl localhost:5000/status?token=12345

CORS
----

By default CORS (cross-origin resource sharing) calls are not allowed. If you want to call your Rasa NLU server from another domain (for example from a training web UI) then you can whitelist that domain by adding it to the config value ``cors_origin``.

.. include:: feedback.inc


