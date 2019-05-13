:desc: Access and configure the HTTP API of Rasa NLU to run
       the nlp library as a server.

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

By default the currently loaded model is used to parse the request.
You can also query against a specific model:

.. code-block:: console

    $ curl -XPOST localhost:5000/parse -d '{"q":"hello there", "model": "<model_XXXXXX>"}'


``POST /train``
^^^^^^^^^^^^^^^

You can post your training data to this endpoint to train a new model.
This request will wait for the server answer: either the model
was trained successfully or the training exited with an error. If the model
is trained successfully a zip file is returned with the trained model.
The configuration of the model should be
posted as the content of the request:

**Using training data in json format**:

.. literalinclude:: ../../sample_configs/config_train_server_json.yml

**Using training data in md format**:

.. literalinclude:: ../../sample_configs/config_train_server_md.yml


Here is an example request showcasing how to send the config to the server
to start the training:

.. code-block:: bash

    $ curl -XPOST -H "Content-Type: application/x-yml" localhost:5000/train \
        -d @sample_configs/config_train_server_md.yml

.. note::

    The request should always be sent as application/x-yml regardless of whether you use json or md for the data format. Do not send json as application/json for example.

.. note::

    The server will automatically generate a name for the trained model. If
    you want to set the name yourself, call the endpoint using
    ``localhost:5000/train?model=my_model_name``

``POST /evaluate``
^^^^^^^^^^^^^^^^^^

You can use this endpoint to evaluate data on a model. The query string
takes a ``model`` (optional). N.b. if you don't specify
a model, the currently loaded model will be selected. If you specify a model, the model should match the currently
loaded one. This endpoint returns some common
sklearn  evaluation metrics (`accuracy <http://scikit-learn
.org/stable/modules/generated/sklearn.metrics.accuracy_score.html#sklearn
.metrics.accuracy_score>`_, `f1 score <http://scikit-learn
.org/stable/modules/generated/sklearn.metrics.f1_score.html>`_,
`precision <http://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_score.html>`_, as well as
a summary `report <http://scikit-learn.org/stable/modules/generated/sklearn
.metrics.classification_report.html>`_) for both intents and entities.

.. code-block:: bash

    $ curl -XPOST localhost:5000/evaluate?model=model_XXXXXX -d @data/examples/rasa/demo-rasa.json | python -mjson.tool

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
            "CRFEntityExtractor": {
                "report": ...,
                "precision": 0.7606987393295268,
                "f1_score": 0.812633994625117,
                "accuracy": 0.8721804511278195
            }
        }
    }


``GET /status``
^^^^^^^^^^^^^^^

This returns the name of the currently loaded model and some information about the worker processes.

.. code-block:: bash

    $ curl localhost:5000/status | python -mjson.tool

    {
        "max_worker_processes": 2,
        "current_worker_processes": 1,
        "loaded_model": "restaurant_bot.tar.gz",
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

``DELETE /models``
^^^^^^^^^^^^^^^^^^

This will unload a model from the server memory

.. code-block:: bash

    $ curl -X DELETE localhost:5000/models?model=model_XXXXXX
    }

``PUT /models``
^^^^^^^^^^^^^^^^^^

This will load a model

.. code-block:: bash

    $ curl -X PUT localhost:5000/models?model=model_XXXXXX


.. include:: feedback.inc
