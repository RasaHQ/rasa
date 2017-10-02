.. _section_http:

Using Rasa Core as a HTTP server
================================

.. note::

    Before you can use the server, you need to define a domain, create training
    data, and train a model. You can then use the trained model for remote code
    execution! See :ref:`tour` for an introduction.

The HTTP api exists to make it easy for non-python projects to use Rasa Core.

Running the server
------------------
You can run a simple http server that handles requests using your
models with

.. code-block:: bash

    $ python -m rasa_core.server -d examples/babi/models/policy/current -u examples/babi/models/nlu/current_py2 -o out.log


Endpoints
---------

``POST /conversation/<cid>/parse``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

You must POST data in this format ``'{"query":"<your text to parse>"}'``,
you can do this with

.. code-block:: bash

    $ curl -XPOST localhost:5005/parse -d '{"query":"hello there"}'


``POST /conversation/<cid>/continue``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

You can post your training data to this endpoint to train a new model.
This request will wait for the server answer: either the model was trained successfully or the training errored.
If you want to name your model to be able to use it during parse requests later on,
you should pass the name ``/train?name=my_model``. Any parameter passed with the query string will be treated as a
configuration parameter of the model, hence you can change all the configuration values listed in the
configuration section by passing in their name and the adjusted value.

.. code-block:: bash

    $ curl -XPOST localhost:5000/train -d @data/examples/rasa/demo-rasa.json


``GET /version``
^^^^^^^^^^^^^^^^

This will return the current version of the Rasa NLU instance.

.. code-block:: bash

    $ curl localhost:5005/version | python -mjson.tool
    {
      "version" : "0.7.2"
    }

.. _section_events_actions:

Events and Action Execution
---------------------------

Instead of writing the actions in python code, you can use the http API to write
the code that should be run in an arbitrary language.