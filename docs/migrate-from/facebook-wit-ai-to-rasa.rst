
:desc: Open source alternative to Facebooks Wit.ai for conversational bots and NLP

.. _facebook-wit-ai-to-rasa:

Rasa as open source alternative to Facebook's Wit.ai - Migration Guide
======================================================================

This guide shows you how to migrate your application built with Facebook's Wit.ai to Rasa. Here are a few reasons why we see developers switching:

* **Faster**: Runs locally - no https requests and server round trips required
* **Customizable**: Tune models and get higher accuracy with your data set
* **Open source**: No risk of vendor lock-in - the Rasa Stack comes with an Apache 2.0 licence and you can use it in commercial projects


.. raw:: html

     In addition, our open source tools allow developers to build contextual AI assistants and manage dialogues with machine learning instead of rules - learn more in <a class="reference external" href="http://blog.rasa.com/a-new-approach-to-conversational-software/" target="_blank">this blog post</a>.


Let's get started with migrating your application from Wit.ai to Rasa:


Step 1: Export your Training Data from Wit.ai
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Navigate to your app's setting page by clicking the **Settings** icon in the upper right corner. Scroll down to **Export your data** and hit the button **Download .zip with your data**.

This will download a file with a ``.zip`` extension. Unzip this file to create a folder. The file you want from your download is called ``expressions.json``


Step 2: Train your Rasa NLU model
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Follow the instructions in the `NLU Quickstart <https://rasa.com/docs/nlu/quickstart/>`_, using your downloaded file as the training data.


Step 3: Modify your app to call your Rasa NLU Server
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Your existing application will have some code to make API requests to Wit.ai.
Modify the API url to point to your Rasa NLU server.
If you are testing this on your development machine, that will be at ``http://localhost:5000``
When you start the Rasa NLU server, you can also pass an ``emulate`` argument:

.. code-block:: bash

    python -m rasa_nlu.server -e wit

By adding this parameter, Rasa NLU's responses will be in the same format as Wit.ai provides,
so that you don't have to modify anything other than the URL in your API call.

|

Join the `Rasa Community Forum <https://forum.rasa.com/>`_ and let us know how your migration went!
