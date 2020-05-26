:desc: Find out how to use only Rasa NLU as a standalone NLU service for your chatbot or virtual assistant. a
 a
.. _using-nlu-only: a
 a
Using NLU Only a
============== a
 a
.. edit-link:: a
 a
 a
If you want to use Rasa only as an NLU component, you can! a
 a
Training NLU-only models a
------------------------ a
 a
To train an NLU model only, run: a
 a
.. code-block:: bash a
 a
   rasa train nlu a
 a
This will look for NLU training data files in the ``data/`` directory a
and saves a trained model in the ``models/`` directory. a
The name of the model will start with ``nlu-``. a
 a
Testing your NLU model on the command line a
------------------------------------------ a
 a
To try out your NLU model on the command line, use the ``rasa shell nlu`` command: a
 a
 a
.. code-block:: bash a
 a
    rasa shell nlu a
 a
 a
This will start the rasa shell and ask you to type in a message to test. a
You can keep typing in as many messages as you like. a
 a
Alternatively, you can leave out the ``nlu`` argument and pass in an nlu-only model directly: a
 a
.. code-block:: bash a
 a
    rasa shell -m models/nlu-20190515-144445.tar.gz a
 a
 a
 a
Running an NLU server a
--------------------- a
 a
To start a server with your NLU model, pass in the model name at runtime: a
 a
.. code-block:: bash a
 a
    rasa run --enable-api -m models/nlu-20190515-144445.tar.gz a
 a
 a
You can then request predictions from your model using the ``/model/parse`` endpoint. a
To do this, run: a
 a
.. code-block:: bash a
 a
   curl localhost:5005/model/parse -d '{"text":"hello"}' a
 a