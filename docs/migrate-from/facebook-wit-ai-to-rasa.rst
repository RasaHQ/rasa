:desc: Open source alternative to Facebook's Wit.ai for conversational bots and NLP a
 a
.. _facebook-wit-ai-to-rasa: a
 a
Rasa as open source alternative to Facebook's Wit.ai - Migration Guide a
====================================================================== a
 a
.. edit-link:: a
 a
This guide shows you how to migrate your application built with Facebook's Wit.ai to Rasa. Here are a few reasons why we see developers switching: a
 a
* **Faster**: Runs locally - no http requests and server round trips required a
* **Customizable**: Tune models and get higher accuracy with your data set a
* **Open source**: No risk of vendor lock-in - Rasa is under the Apache 2.0 licence and you can use it in commercial projects a
 a
 a
.. raw:: html a
 a
     In addition, our open source tools allow developers to build contextual AI assistants and manage dialogues with machine learning instead of rules - learn more in <a class="reference external" href="http://blog.rasa.com/a-new-approach-to-conversational-software/" target="_blank">this blog post</a>. a
 a
 a
Let's get started with migrating your application from Wit.ai to Rasa: a
 a
 a
Step 1: Export your Training Data from Wit.ai a
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ a
 a
Navigate to your app's setting page by clicking the **Settings** icon in the upper right corner. Scroll down to **Export your data** and hit the button **Download .zip with your data**. a
 a
This will download a file with a ``.zip`` extension. Unzip this file to create a folder. The file you want from your download is called ``expressions.json`` a
 a
Step 2: Create a Rasa Project a
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ a
 a
To create a Rasa project, run: a
 a
.. code-block:: bash a
 a
   rasa init a
 a
This will create a directory called ``data``.  a
Remove the files in this directory, and a
move the expressions.json file into this directory. a
 a
.. code-block:: bash a
 a
   rm -r data/* a
   mv /path/to/expressions.json data/ a
 a
 a
 a
Step 3: Train your NLU model a
^^^^^^^^^^^^^^^^^^^^^^^^^^^^ a
 a
To train a model using your Wit data, run: a
 a
.. code-block:: bash a
 a
    rasa train nlu a
 a
Step 4: Test your NLU model a
^^^^^^^^^^^^^^^^^^^^^^^^^^^ a
 a
Let's see how your NLU model will interpret some test messages. a
To start a testing session, run: a
 a
.. code-block:: bash a
 a
   rasa shell nlu a
 a
This will prompt your for input. a
Type a test message and press 'Enter'. a
The output of your NLU model will be printed to the screen. a
You can keep entering messages and test as many as you like. a
Press 'control + C' to quit. a
 a
 a
Step 5: Start a Server with your NLU Model a
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ a
 a
To start a server with your NLU model, run: a
 a
.. code-block:: bash a
 a
   rasa run nlu a
 a
This will start a server listening on port 5005. a
 a
To send a request to the server, run: a
 a
.. copyable:: a
 a
   curl 'localhost:5005/model/parse?emulation_mode=wit' -d '{"text": "hello"}' a
 a
The ``emulation_mode`` parameter tells Rasa that you want your json a
response to have the same format as you would get from wit.ai. a
You can also leave it out to get the result in the usual Rasa format. a
 a
 a
Join the `Rasa Community Forum <https://forum.rasa.com/>`_ and let us know how your migration went! a
 a