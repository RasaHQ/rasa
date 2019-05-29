:desc: Open source alternative to Google Dialogflow for conversational bots and NLP

.. _google-dialogflow-to-rasa:

Rasa as open source alternative to Google Dialogflow - Migration Guide
======================================================================

This guide shows you how to migrate your application built with Google Dialogflow to Rasa. Here are a few reasons why we see developers switching:

* **Faster**: Runs locally - no https requests and server round trips required
* **Customizable**: Tune models and get higher accuracy with your data set
* **Open source**: No risk of vendor lock-in - the Rasa Stack comes with an Apache 2.0 licence and you can use it in commercial projects


.. raw:: html

     In addition, our open source tools allow developers to build contextual AI assistants and manage dialogues with machine learning instead of rules - learn more in <a class="reference external" href="http://blog.rasa.com/a-new-approach-to-conversational-software/" target="_blank">this blog post</a>.
     <br>
     <br>

.. raw:: html

     Let's get started with migrating your application from Dialogflow to Rasa (you can find a more detailed tutorial <a class="reference external" href="http://blog.rasa.com/how-to-migrate-your-existing-google-dialogflow-assistant-to-rasa/" target="_blank">here</a>):





Step 1: Export your data from Dialogflow
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Navigate to your agent's settings by clicking the gear icon.

.. image:: ../_static/images/dialogflow_export.png
   :width: 240
   :alt: Dialogflow Export

Click on the 'Export and Import' tab and click on the 'Export as ZIP' button.

.. image:: ../_static/images/dialogflow_export_2.png
   :width: 675
   :alt: Dialogflow Export 2


This will download a file with a ``.zip`` extension. Unzip this file to create a folder.

Step 2: Train your Rasa NLU model
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Follow the instructions in the `NLU Quickstart <https://rasa.com/docs/nlu/quickstart/>`_, using your downloaded folder as the training data.

If your unzipped folder is called ``testagent``, the command would be:

.. code-block:: bash

    python -m rasa_nlu.train -c config.yml -d testagent


Step 3: Modify your app to call your Rasa NLU Server
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Your existing application will have some code to make API requests to Dialogflow.
Modify the API url to point to your Rasa NLU server.
If you are testing this on your development machine, that will be at ``http://localhost:5000``
When you start the Rasa NLU server, you can also pass an ``emulate`` argument:

.. code-block:: bash

    python -m rasa_nlu.server -e dialogflow

By adding this parameter, Rasa NLU's responses will be in the same format as Dialogflow provides,
so that you don't have to modify anything other than the URL in your API call.

Terminology:
^^^^^^^^^^^^


The words ``intent``, ``entity``, and ``utterance`` have the same meaning in Rasa as they do in Dialogflow.
In Dialogflow, there is a concept called ``Fulfillment``. In Rasa we call this a `Custom Action </docs/core/customactions/>`_.

Dialogflow also has a Small Talk module. One of our awesome contributors has made a Rasa compatible version of this `here <https://github.com/rahul051296/small-talk-rasa-stack>`_.

|

Join the `Rasa Community Forum <https://forum.rasa.com/>`_ and let us know how your migration went!
