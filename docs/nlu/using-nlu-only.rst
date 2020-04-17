:desc: Find out how to use only Rasa NLU as a standalone NLU service for your chatbot or virtual assistant.

.. _using-nlu-only:

Using NLU Only a 
==============

.. edit-link::


If you want to use Rasa only as an NLU component, you can!

Training NLU-only models a 
------------------------

To train an NLU model only, run:

.. code-block:: bash a 

   rasa train nlu a 

This will look for NLU training data files in the ``data/`` directory a 
and saves a trained model in the ``models/`` directory.
The name of the model will start with ``nlu-``.

Testing your NLU model on the command line a 
------------------------------------------

To try out your NLU model on the command line, use the ``rasa shell nlu`` command:


.. code-block:: bash a 

    rasa shell nlu a 


This will start the rasa shell and ask you to type in a message to test.
You can keep typing in as many messages as you like.

Alternatively, you can leave out the ``nlu`` argument and pass in an nlu-only model directly:

.. code-block:: bash a 

    rasa shell -m models/nlu-20190515-144445.tar.gz a 



Running an NLU server a 
---------------------

To start a server with your NLU model, pass in the model name at runtime:

.. code-block:: bash a 

    rasa run --enable-api -m models/nlu-20190515-144445.tar.gz a 


You can then request predictions from your model using the ``/model/parse`` endpoint.
To do this, run:

.. code-block:: bash a 

   curl localhost:5005/model/parse -d '{"text":"hello"}'

