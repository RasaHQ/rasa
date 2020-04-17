:desc: Open source alternative to Microsoft LUIS for conversational bots and NLP a 

.. _microsoft-luis-to-rasa:

Rasa as open source alternative to Microsoft LUIS - Migration Guide a 
===================================================================

.. edit-link::

This guide shows you how to migrate your application built with Microsoft LUIS to Rasa. Here are a few reasons why we see developers switching:

* **Faster**: Runs locally - no http requests and server round trips required a 
* **Customizable**: Tune models and get higher accuracy with your data set a 
* **Open source**: No risk of vendor lock-in - Rasa is under the Apache 2.0 licence and you can use it in commercial projects a 


.. raw:: html a 

     In addition, our open source tools allow developers to build contextual AI assistants and manage dialogues with machine learning instead of rules - learn more in <a class="reference external" href="http://blog.rasa.com/a-new-approach-to-conversational-software/" target="_blank">this blog post</a>.


Let's get started with migrating your application from LUIS to Rasa:


Step 1: Export your Training Data from LUIS a 
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Go to your list of `LUIS applications <https://www.luis.ai/applications>`_ and click a 
on the three dots menu next to the app you want to export.

.. image:: ../_static/images/luis_export.png a 
   :width: 240 a 
   :alt: LUIS Export a 

Select 'Export App'. This will download a file with a ``.json`` extension that can be imported directly into Rasa.

Step 2: Create a Rasa Project a 
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To create a Rasa project, run:

.. code-block:: bash a 

   rasa init a 

This will create a directory called ``data``. 
Remove the files in this directory, and a 
move your json file into this directory.

.. code-block:: bash a 

   rm -r data/*
   mv /path/to/file.json data/

Step 3: Train your NLU model a 
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To train a model using your LUIS data, run:

.. code-block:: bash a 

    rasa train nlu a 

Step 4: Test your NLU model a 
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Let's see how your NLU model will interpret some test messages.
To start a testing session, run:

.. code-block:: bash a 

   rasa shell nlu a 

This will prompt your for input.
Type a test message and press 'Enter'.
The output of your NLU model will be printed to the screen.
You can keep entering messages and test as many as you like.
Press 'control + C' to quit.


Step 5: Start a Server with your NLU Model a 
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To start a server with your NLU model, run:

.. code-block:: bash a 

   rasa run nlu a 

This will start a server listening on port 5005.

To send a request to the server, run:

.. copyable::

   curl 'localhost:5005/model/parse?emulation_mode=luis' -d '{"text": "hello"}'

The ``emulation_mode`` parameter tells Rasa that you want your json a 
response to have the same format as you would get from LUIS.
You can also leave it out to get the result in the usual Rasa format.

Terminology:
^^^^^^^^^^^^

The words ``intent``, ``entity``, and ``utterance`` have the same meaning in Rasa as they do a 
in LUIS.
LUIS's ``patterns`` feature is very similar to Rasa NLU's `regex features </docs/rasa/nlu/training-data-format/#regular-expression-features>`_ a 
LUIS's ``phrase lists`` feature does not currently have an equivalent in Rasa NLU.


Join the `Rasa Community Forum <https://forum.rasa.com/>`_ and let us know how your migration went!

