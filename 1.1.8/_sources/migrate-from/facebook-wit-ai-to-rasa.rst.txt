
:desc: Open source alternative to Facebook's Wit.ai for conversational bots and NLP

.. _facebook-wit-ai-to-rasa:

Rasa as open source alternative to Facebook's Wit.ai - Migration Guide
======================================================================

This guide shows you how to migrate your application built with Facebook's Wit.ai to Rasa. Here are a few reasons why we see developers switching:

* **Faster**: Runs locally - no http requests and server round trips required
* **Customizable**: Tune models and get higher accuracy with your data set
* **Open source**: No risk of vendor lock-in - Rasa is under the Apache 2.0 licence and you can use it in commercial projects


.. raw:: html

     In addition, our open source tools allow developers to build contextual AI assistants and manage dialogues with machine learning instead of rules - learn more in <a class="reference external" href="http://blog.rasa.com/a-new-approach-to-conversational-software/" target="_blank">this blog post</a>.


Let's get started with migrating your application from Wit.ai to Rasa:


Step 1: Export your Training Data from Wit.ai
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Navigate to your app's setting page by clicking the **Settings** icon in the upper right corner. Scroll down to **Export your data** and hit the button **Download .zip with your data**.

This will download a file with a ``.zip`` extension. Unzip this file to create a folder. The file you want from your download is called ``expressions.json``

Step 2: Create a Rasa Project
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To create a Rasa project, run:

.. code-block:: bash

   rasa init

This will create a directory called ``data``. 
Remove the files in this directory, and
move the expressions.json file into this directory.

.. code-block:: bash

   rm -r data/*
   mv /path/to/expressions.json data/



Step 3: Train your NLU model
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To train a model using your Wit data, run:

.. code-block:: bash

    rasa train nlu

Step 4: Test your NLU model
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Let's see how your NLU model will interpret some test messages.
To start a testing session, run:

.. code-block:: bash

   rasa shell nlu

This will prompt your for input.
Type a test message and press 'Enter'.
The output of your NLU model will be printed to the screen.
You can keep entering messages and test as many as you like.
Press 'control + C' to quit.


Step 5: Start a Server with your NLU Model
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To start a server with your NLU model, run:

.. code-block:: bash

   rasa run nlu

This will start a server listening on port 5005.

To send a request to the server, run:

.. copyable::

   curl 'localhost:5005/model/parse?emulation_mode=wit' -d '{"text": "hello"}'

The ``emulation_mode`` parameter tells Rasa that you want your json
response to have the same format as you would get from wit.ai.
You can also leave it out to get the result in the usual Rasa format.


Join the `Rasa Community Forum <https://forum.rasa.com/>`_ and let us know how your migration went!
