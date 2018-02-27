.. _section_http:

Using rasa NLU as a HTTP server
===============================

.. note:: Before you can use the server, you need to train a model! See :ref:`training_your_model`

The HTTP api exists to make it easy for non-python projects to use rasa NLU, and to make it trivial for projects currently using wit/LUIS/Dialogflow to try it out.

Running the server
------------------
You can run a simple http server that handles requests using your projects with :

.. code-block:: bash

    $ python -m rasa_nlu.server -c sample_configs/config_spacy.json

The server will look for existing projects under the folder defined by the ``path`` parameter in the configuration.
By default a project will load the latest trained model.


Emulation
---------
rasa NLU can 'emulate' any of these three services by making the ``/parse`` endpoint compatible with your existing code.
To activate this, either add ``'emulate' : 'luis'`` to your config file or run the server with ``-e luis``.
For example, if you would normally send your text to be parsed to LUIS, you would make a ``GET`` request to

``https://api.projectoxford.ai/luis/v2.0/apps/<app-id>?q=hello%20there``

in luis emulation mode you can call rasa by just sending this request to 

``http://localhost:5000/parse?q=hello%20there``

any extra query params are ignored by rasa, so you can safely send them along. 


Endpoints
---------

``POST /parse`` (no emulation)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

You must POST data in this format ``'{"q":"<your text to parse>"}'``, you can do this with

.. code-block:: bash

    $ curl -XPOST localhost:5000/parse -d '{"q":"hello there"}'

By default, when the project is not specified in the query, the ``"default"`` one will be used.
You can (should) specify the project you want to use in your query :

.. code-block:: bash

    $ curl -XPOST localhost:5000/parse -d '{"q":"hello there", "project": "my_restaurant_search_bot"}

By default the latest trained model for the project will be loaded. You can also query against a specific model for a project :

.. code-block:: bash

    $ curl -XPOST localhost:5000/parse -d '{"q":"hello there", "project": "my_restaurant_search_bot", "model": <model_XXXXXX>}


``POST /train``
^^^^^^^^^^^^^^^

You can post your training data to this endpoint to train a new model for a project.
This request will wait for the server answer: either the model was trained successfully or the training errored.
Using the HTTP server, you must specify the project you want to train a new model for to be able to use it during parse requests later on :
``/train?project=my_project``. Any parameter passed with the query string will be treated as a
configuration parameter of the model, hence you can change all the configuration values listed in the
configuration section by passing in their name and the adjusted value.

.. code-block:: bash

    $ curl -XPOST localhost:5000/train?project=my_project -d @data/examples/rasa/demo-rasa.json

You cannot send a training request for a project already training a new model (see below).


``POST /evaluate``
^^^^^^^^^^^^^^^^^^

You can use this endpoint to evaluate data on a model. The query string
takes the ``project`` (required) and a ``model`` (optional). You must
specify the project in which the model is located. N.b. if you don't specify
a model, the latest one will be selected. This endpoint returns some common
sklearn  evaluation metrics (`accuracy <http://scikit-learn
.org/stable/modules/generated/sklearn.metrics.accuracy_score.html#sklearn
.metrics.accuracy_score>`_, `f1 score <http://scikit-learn
.org/stable/modules/generated/sklearn.metrics.f1_score.html>`_,
`precision <http://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_score.html>`_, as well as
a summary `report <http://scikit-learn.org/stable/modules/generated/sklearn
.metrics.classification_report.html>`_).

.. code-block:: bash

    $ curl -XPOST localhost:5000/evaluate?project=my_project&model=model_XXXXXX -d @data/examples/rasa/demo-rasa.json | python -mjson.tool

    {
        "accuracy": 0.19047619047619047,
        "f1_score": 0.06095238095238095,
        "precision": 0.036281179138321996,
        "predictions": [
            {
                "intent": "greet",
                "predicted": "greet",
                "text": "hey",
                "confidence": 1.0
            },
            ...,
        ]
        "report": ...
    }


``GET /status``
^^^^^^^^^^^^^^^

This returns all the currently available projects, their status (``training`` or ``ready``) and their models loaded in memory.
also returns a list of available projects the server can use to fulfill ``/parse`` requests.

.. code-block:: bash

    $ curl localhost:5000/status | python -mjson.tool
    
    {
      "available_projects": {
        "my_restaurant_search_bot" : {
          "status" : "ready",
          "available_models" : [
            <model_XXXXXX>,
            <model_XXXXXX>
          ]
        }
      }
    }

``GET /version``
^^^^^^^^^^^^^^^^

This will return the current version of the Rasa NLU instance.

.. code-block:: bash

    $ curl localhost:5000/version | python -mjson.tool
    {
      "version" : "0.8.2"
    }

    
``GET /config``
^^^^^^^^^^^^^^^

This will return the currently running configuration of the Rasa NLU instance.

.. code-block:: bash

    $ curl localhost:5000/config | python -mjson.tool
    {
        "config": "/app/rasa_shared/config_mitie.json",
        "data": "/app/rasa_nlu/data/examples/rasa/demo-rasa.json",
        "duckling_dimensions": null,
        "emulate": null,
        ...
      }

.. _section_auth:

Authorization
-------------
To protect your server, you can specify a token in your rasa NLU configuration, e.g. by adding ``"token" : "12345"`` to your config file, or by setting the ``RASA_TOKEN`` environment variable.
If set, this token must be passed as a query parameter in all requests, e.g. :

.. code-block:: bash

    $ curl localhost:5000/status?token=12345

On default CORS (cross-origin resource sharing) calls are not allowed. If you want to call your rasa NLU server from another domain (for example from a training web UI) then you can whitelist that domain by adding it to the config value ``cors_origin``.


.. _section_http_config:

Serving Multiple Apps
---------------------

Depending on your choice of backend, rasa NLU can use quite a lot of memory.
So if you are serving multiple models in production, you want to serve these
from the same process & avoid duplicating the memory load.

.. note::
Although this saves the backend from loading the same backend twice, it still needs to load one set of
    word vectors (which make up most of the memory consumption) per language and backend.

As stated previously, Rasa NLU naturally handles serving multiple apps : by default the server will load all projects found
under the ``path`` directory defined in the configuration. The file structure under ``path directory`` is as follows :

- <path>
 - <project_A>
  - <model_XXXXXX>
  - <model_XXXXXX>
   ...
 - <project_B>
  - <model_XXXXXX>
   ...
  ...


So you can specify which one to use in your ``/parse`` requests:

.. code-block:: console

    $ curl 'localhost:5000/parse?q=hello&project=my_restaurant_search_bot'

or

.. code-block:: console

    $ curl -XPOST localhost:5000/parse -d '{"q":"I am looking for Chinese food", "project":"my_restaurant_search_bot"}'

You can also specify the model you want to use for a given project, the default used being the latest trained :

.. code-block:: console

    $ curl -XPOST localhost:5000/parse -d '{"q":"I am looking for Chinese food", "project":"my_restaurant_search_bot", "model":<model_XXXXXX>}'

If no project is to be found by the server under the ``path`` directory, a ``"default"`` one will be used, using a simple fallback model.
