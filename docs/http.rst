:desc: The Rasa NLU REST API
.. _section_http:

HTTP API
========

.. contents::

Endpoints
---------

``POST /parse``
^^^^^^^^^^^^^^^

You must POST data in this format ``'{"q":"<your text to parse>"}'``,
you can do this with

.. code-block:: bash

    $ curl -XPOST localhost:5000/parse -d '{"q":"hello there"}'

By default, when the project is not specified in the query, the
``"default"`` one will be used.
You can (should) specify the project you want to use in your query :

.. code-block:: console

    $ curl -XPOST localhost:5000/parse -d '{"q":"hello there", "project": "my_restaurant_search_bot"}'

By default the latest trained model for the project will be loaded.
You can also query against a specific model for a project :

.. code-block:: console

    $ curl -XPOST localhost:5000/parse -d '{"q":"hello there", "project": "my_restaurant_search_bot", "model": "<model_XXXXXX>"}'


``POST /train``
^^^^^^^^^^^^^^^

You can post your training data to this endpoint to train a new model for a project.
This request will wait for the server answer: either the model
was trained successfully or the training exited with an error.
Using the HTTP server, you must specify the project you want to train a
new model for to be able to use it during parse requests later on :
``/train?project=my_project``. The configuration of the model should be
posted as the content of the request:

**Using training data in json format**:

.. literalinclude:: ../sample_configs/config_train_server_json.yml

**Using training data in md format**:

.. literalinclude:: ../sample_configs/config_train_server_md.yml


Here is an example request showcasing how to send the config to the server
to start the training:

.. code-block:: bash

    $ curl -XPOST -H "Content-Type: application/x-yml" localhost:5000/train?project=my_project \
        -d @sample_configs/config_train_server_md.yml
        
.. note::

    The request should always be sent as application/x-yml regardless of wether you use json or md for the data format. Do not send json as application/json for example.

.. note::

    You cannot send a training request for a project
    already training a new model (see below).

.. note::

    The server will automatically generate a name for the trained model. If
    you want to set the name yourself, call the endpoint using
    ``localhost:5000/train?project=my_project&model=my_model_name``

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
.metrics.classification_report.html>`_) for both intents and entities.

.. code-block:: bash

    $ curl -XPOST localhost:5000/evaluate?project=my_project&model=model_XXXXXX -d @data/examples/rasa/demo-rasa.json | python -mjson.tool

    {
        "intent_evaluation": {
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
        },
        "entity_evaluation": {
            "ner_crf": {
                "report": ...,
                "precision": 0.7606987393295268,
                "f1_score": 0.812633994625117,
                "accuracy": 0.8721804511278195
            }
        }
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

This will return the current version of the Rasa NLU instance, as well as the minimum model version required for loading models.

.. code-block:: bash

    $ curl localhost:5000/version | python -mjson.tool
    {
      "version" : "0.13.0",
      "minimum_compatible_version": "0.13.0"
    }

    
``GET /config``
^^^^^^^^^^^^^^^

This will return the default model configuration of the Rasa NLU instance.

.. code-block:: bash

    $ curl localhost:5000/config | python -mjson.tool
    {
        "config": "/app/rasa_shared/config_mitie.json",
        "data": "/app/rasa_nlu/data/examples/rasa/demo-rasa.json",
        "duckling_dimensions": null,
        "emulate": null,
        ...
      }

``DELETE /models``
^^^^^^^^^^^^^^^^^^

This will unload a model from the server memory

.. code-block:: bash

    $ curl -X DELETE localhost:5000/models?project=my_restaurant_search_bot&model=model_XXXXXX


.. include:: feedback.inc
	

