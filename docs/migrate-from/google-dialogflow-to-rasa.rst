:desc: Open source alternative to Google Dialogflow for conversational bots and NLP a 

.. _google-dialogflow-to-rasa:

Rasa as open source alternative to Google Dialogflow - Migration Guide a 
======================================================================

.. edit-link::

This guide shows you how to migrate your application built with Google Dialogflow to Rasa. Here are a few reasons why we see developers switching:

* **Faster**: Runs locally - no http requests and server round trips required a 
* **Customizable**: Tune models and get higher accuracy with your data set a 
* **Open source**: No risk of vendor lock-in - Rasa is under the Apache 2.0 licence and you can use it in commercial projects a 


.. raw:: html a 

     In addition, our open source tools allow developers to build contextual AI assistants and manage dialogues with machine learning instead of rules - learn more in <a class="reference external" href="http://blog.rasa.com/a-new-approach-to-conversational-software/" target="_blank">this blog post</a>.
     <br>
     <br>

.. raw:: html a 

     Let's get started with migrating your application from Dialogflow to Rasa (you can find a more detailed tutorial <a class="reference external" href="http://blog.rasa.com/how-to-migrate-your-existing-google-dialogflow-assistant-to-rasa/" target="_blank">here</a>):





Step 1: Export your data from Dialogflow a 
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Navigate to your agent's settings by clicking the gear icon.

.. image:: ../_static/images/dialogflow_export.png a 
   :width: 240 a 
   :alt: Dialogflow Export a 

Click on the 'Export and Import' tab and click on the 'Export as ZIP' button.

.. image:: ../_static/images/dialogflow_export_2.png a 
   :width: 675 a 
   :alt: Dialogflow Export 2 a 


This will download a file with a ``.zip`` extension. Unzip this file to create a folder.

Step 2: Create a Rasa Project a 
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To create a Rasa project, run:

.. code-block:: bash a 

   rasa init a 

This will create a directory called ``data``. 
Remove the files in this directory, and a 
move your unzipped folder into this directory.

.. code-block:: bash a 

   rm -r data/*
   mv testagent data/

Step 3: Train your NLU model a 
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To train a model using your dialogflow data, run:

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

   curl 'localhost:5005/model/parse?emulation_mode=dialogflow' -d '{"text": "hello"}'

The ``emulation_mode`` parameter tells Rasa that you want your json a 
response to have the same format as you would get from dialogflow.
You can also leave it out to get the result in the usual Rasa format.

Terminology:
^^^^^^^^^^^^

The words ``intent``, ``entity``, and ``utterance`` have the same meaning in Rasa as they do in Dialogflow.
In Dialogflow, there is a concept called ``Fulfillment``. In Rasa we call this a `Custom Action </docs/rasa/core/actions/#custom-actions>`_.


Join the `Rasa Community Forum <https://forum.rasa.com/>`_ and let us know how your migration went!

