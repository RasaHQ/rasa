:desc: Find out how to use only Rasa NLU as a standalone NLU service for your chatbot or virtual assistant.

.. _using-nlu-only:

Using NLU Only
==============

.. edit-link::


If you want to use Rasa only as an NLU component, you can!

Training NLU-only models
------------------------

To train an NLU model only, run:

.. code-block:: bash

   rasa train nlu

This will look for NLU training data files in the ``data/`` directory
and saves a trained model in the ``models/`` directory.
The name of the model will start with ``nlu-``.

Testing your NLU model on the command line
------------------------------------------

To try out your NLU model on the command line, use the ``rasa shell nlu`` command:


.. code-block:: bash

    rasa shell nlu


This will start the rasa shell and ask you to type in a message to test.
You can keep typing in as many messages as you like.

Alternatively, you can leave out the ``nlu`` argument and pass in an nlu-only model directly:

.. code-block:: bash

    rasa shell -m models/nlu-20190515-144445.tar.gz



Running an NLU server
---------------------

To start a server with your NLU model, pass in the model name at runtime:

.. code-block:: bash

    rasa run --enable-api -m models/nlu-20190515-144445.tar.gz


You can then request predictions from your model using the ``/model/parse`` endpoint.
To do this, run:

.. code-block:: bash

   curl localhost:5005/model/parse -d '{"text":"hello"}'
