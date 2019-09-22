:desc: This tutorial will show you the different parts needed to build a
       chatbot or AI assistant using open source Rasa.

.. _rasa-tutorial:

Rasa Tutorial
=============

.. edit-link::

This page explains the basics of building an assistant with Rasa and
shows the structure of a Rasa project. You can test it out right here without
installing anything.
You can also :ref:`install Rasa <installation>` and follow along in your command line.

.. raw:: html

    The <a style="text-decoration: none" href="https://rasa.com/docs/rasa/glossary">glossary</a> contains an overview of the most common terms you’ll see in the Rasa documentation.



.. contents::
   :local:


In this tutorial, you will build a simple, friendly assistant which will ask how you're doing
and send you a fun picture to cheer you up if you are sad.

.. image:: /_static/images/mood_bot.png


1. Create a New Project
^^^^^^^^^^^^^^^^^^^^^^^

The first step is to create a new Rasa project. To do this, run:



.. runnable::

   rasa init --no-prompt


The ``rasa init`` command creates all the files that a Rasa project needs and
trains a simple bot on some sample data.
If you leave out the ``--no-prompt`` flag you will be asked some questions about
how you want your project to be set up.

This creates the following files:


+-------------------------------+--------------------------------------------------------+
| ``__init__.py``               | an empty file that helps python find your actions      |
+-------------------------------+--------------------------------------------------------+
| ``actions.py``                | code for your custom actions                           |
+-------------------------------+--------------------------------------------------------+
| ``config.yml`` '*'            | configuration of your NLU and Core models              |
+-------------------------------+--------------------------------------------------------+
| ``credentials.yml``           | details for connecting to other services               |
+-------------------------------+--------------------------------------------------------+
| ``data/nlu.md`` '*'           | your NLU training data                                 |
+-------------------------------+--------------------------------------------------------+
| ``data/stories.md`` '*'       | your stories                                           |
+-------------------------------+--------------------------------------------------------+
| ``domain.yml`` '*'            | your assistant's domain                                |
+-------------------------------+--------------------------------------------------------+
| ``endpoints.yml``             | details for connecting to channels like fb messenger   |
+-------------------------------+--------------------------------------------------------+
| ``models/<timestamp>.tar.gz`` | your initial model                                     |
+-------------------------------+--------------------------------------------------------+



The most important files are marked with a '*'.
You will learn about all of these in this tutorial.


2. View Your NLU Training Data
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The first piece of a Rasa assistant is an NLU model.
NLU stands for Natural Language Understanding, which means turning
user messages into structured data. To do this with Rasa,
you provide training examples that show how Rasa should understand
user messages, and then train a model by showing it those examples.

Run the code cell below to see the NLU training data created by
the ``rasa init`` command:


.. runnable::

   cat data/nlu.md




The lines starting with ``##`` define the names of your ``intents``, which
are groups of messages with the same meaning. Rasa's job will be to
predict the correct intent when your users send new, unseen messages to
your assistant. You can find all the details of the data format in :ref:`training-data-format`.

.. _model-configuration:

3. Define Your Model Configuration
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The configuration file defines the NLU and Core components that your model
will use. In this example, your NLU model will use the
``supervised_embeddings`` pipeline. You can learn about the different NLU pipelines
:ref:`here <choosing-a-pipeline>`.

Let's take a look at your model configuration file.

.. runnable::

   cat config.yml



The ``language`` and ``pipeline`` keys specify how the NLU model should be built.
The ``policies`` key defines the :ref:`policies <policies>` that the Core model will use.



4. Write Your First Stories
^^^^^^^^^^^^^^^^^^^^^^^^^^^

At this stage, you will teach your assistant how to respond to your messages.
This is called dialogue management, and is handled by your Core model.

Core models learn from real conversational data in the form of training "stories".
A story is a real conversation between a user and an assistant.
Lines with intents and entities reflect the user's input and action names show what the
assistant should do in response.

Below is an example of a simple conversation.
The user says hello, and the assistant says hello back.
This is how it looks as a story:

.. code-block:: story

   ## story1
   * greet
      - utter_greet


You can see the full details in :ref:`stories`.

Lines that start with ``-`` are actions taken by the assistant.
In this tutorial, all of our actions are messages sent back to the user,
like ``utter_greet``, but in general, an action can do anything,
including calling an API and interacting with the outside world.

Run the command below to view the example stories inside the file ``data/stories.md``:


.. runnable::

   cat data/stories.md



5. Define a Domain
^^^^^^^^^^^^^^^^^^

The next thing we need to do is define a :ref:`Domain <domains>`.
The domain defines the universe your assistant lives in: what user inputs it
should expect to get, what actions it should be able to predict, how to
respond, and what information to store.
The domain for our assistant is saved in a
file called ``domain.yml``:



.. runnable::

   cat domain.yml



So what do the different parts mean?


+---------------+-------------------------------------------------------------+
| ``intents``   | things you expect users to say                              |
+---------------+-------------------------------------------------------------+
| ``actions``   | things your assistant can do and say                        |
+---------------+-------------------------------------------------------------+
| ``templates`` | template strings for the things your assistant can say      |
+---------------+-------------------------------------------------------------+


**How does this fit together?**
Rasa Core's job is to choose the right action to execute at each step
of the conversation. In this case, our actions simply send a message to the user.
These simple utterance actions are the ``actions`` in the domain that start
with ``utter_``. The assistant will respond with a message based on a template
from the ``templates`` section. See :ref:`custom-actions`
to build actions that do more than just send a message.



6. Train a Model
^^^^^^^^^^^^^^^^

Anytime we add new NLU or Core data, or update the domain or configuration, we
need to re-train a neural network on our example stories and NLU data.
To do this, run the command below. This command will call the Rasa Core and NLU train
functions and store the trained model
into the ``models/`` directory. The command will automatically only retrain the
different model parts if something has changed in their data or configuration.



.. runnable::

   rasa train

   echo "Finished training."



The ``rasa train`` command will look for both NLU and Core data and will train a combined model.


7. Talk to Your Assistant
^^^^^^^^^^^^^^^^^^^^^^^^^

Congratulations! 🚀 You just built an assistant
powered entirely by machine learning.

The next step is to try it out!
If you're following this tutorial on your local machine, start talking to your
assistant by running:

.. code-block:: bash

   rasa shell


Next Steps
^^^^^^^^^^

Now that you've built your first Rasa bot it's time to learn about
some more advanced Rasa features.

- Learn how to implement business logic using :ref:`forms <forms>`
- Learn how to integrate other APIs using :ref:`custom actions <actions>`
- Learn how to connect your bot to different :ref:`messaging apps <messaging-and-voice-channels>`
- Learn about customising the :ref:`components <components>` in your NLU pipeline
- Read about custom and built-in :ref:`entities <entity-extraction>`

You can also use Rasa X to collect more conversations
and improve your assistant:

.. button::
   :text: Try Rasa X
   :link: ../../../rasa-x/

.. juniper::
   :language: bash
