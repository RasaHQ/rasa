:desc: This tutorial will show you the different parts needed to build a
       bot using open source Rasa Core.

.. _project-structure:

Project Structure
=================

This page explains the basics of building an assistant with Rasa and
shows the structure of a rasa project.


.. contents::
   :local:


**Goal**

You will build a friendly chatbot which will ask you how you're doing
and send you a fun picture to cheer you up if you are sad.

.. image:: /_static/images/mood_bot.png


1. Create a New Project
^^^^^^^^^^^^^^^^^^^^^^^


.. runnable::
   :description: stack-init

   rasa init --no-prompt


Let's take a look at the files this created:


.. runnable::
   :description: stack-ls

   ls -1

The training data files are inside the ``./data`` directory.

2. Create NLU examples
^^^^^^^^^^^^^^^^^^^^^^

The first piece of a Rasa assistant is an NLU model.
NLU stands for Natural Language Understanding and means turning
user messages into structured data. To do this with Rasa,
you provide training examples that show how Rasa should understand
user messages, and then train a model by showing it those examples. 

Run the code cell below to see the NLU training data created by 
the ``rasa init`` command:

.. runnable::
   :description: stack-cat-nlu

   cat data/nlu.md

The lines starting with ``##`` define the names of your ``intents``, which
are groups of messages with the same meaning. Rasa's job will be to
predict the correct intent when your users send new, unseen messages to
your assistant. You can find all the details of the data format in :ref:`nlu-data-format`.

.. _model-configuration:

3. Define your model configuration
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The configuration file defines the NLU and Core components that your model
will use. In this example, you will use the
predefined ``supervised_embeddings`` pipeline. You can learn all about NLU pipelines
`here <https://rasa.com/docs/nlu/choosing_pipeline/>`_.

Let's take a look at your model configuration file.
The ``pipeline`` and ``language`` keys specify how the NLU model should be built.
You can read more about this in :ref:`choosing_pipeline`
The ``policies`` key defines the :ref:`policies` that the Core model will use.

.. runnable::
   :description: stack-cat-config

   cat config.yml


4. Write Your First Stories
^^^^^^^^^^^^^^^^^^^^^^^^^^^

At this stage, you will teach your assistant to respond to your messages.
Dialogue is handled by Rasa's ``Core`` module.

Rasa Core models learn from real conversational data in the form of training "stories".
A story is a real conversation between a user and a bot. 
Lines with intents and entities reflect the user's input and action names show what the
assistant should do in response.

Below is an example of a simple conversation. 
The user says hello, and the assistant says hello back.
This is how it looks as a story:

.. code-block:: story

   ## story1
   * greet
      - utter_greet


You can see the full details in :ref:`stories-data-format`

Lines that start with ``-`` are actions taken by the assistant.
In this case, all of our actions are messages sent back to the user,
like ``utter_greet``, but in general, an action can do anything,
including calling an API and interacting with the outside world.

Run the cell below to show the example stories inside the file ``data/stories.md`` :

.. runnable::
   :description: core-cat-stories

   cat data/stories.md


6. Define a Domain
^^^^^^^^^^^^^^^^^^

The next thing we need to do is define a ``Domain``.
The domain defines the universe your bot lives in - what user inputs it
should expect to get, what actions it should be able to predict, how to
respond and what information to store.
Here is the domain for our assistant, it's saved in a 
file called ``domain.yml``:

.. runnable::
   :description: stack-cat-domain

   cat domain.yml


So what do the different parts mean?


+---------------+-------------------------------------------------------------+
| ``intents``   | things you expect users to say. See                         |
|               | `Rasa NLU <https://rasa.com/docs/nlu/>`_                    |
+---------------+-------------------------------------------------------------+
| ``actions``   | things your bot can do and say                              |
+---------------+-------------------------------------------------------------+
| ``templates`` | template strings for the things your bot can say            |
+---------------+-------------------------------------------------------------+


**How does this fit together?**
Rasa Core's job is to choose the right action to execute at each step
of the conversation. Simple actions are sending a message to a user.
These simple actions are the ``actions`` in the domain, which start
with ``utter_``. They will respond with a message based on a template
from the ``templates`` section. See `Custom Actions <https://rasa.com/docs/core/customactions/>`_ for how to build
 actions that do more than just send a message.



7. Train a Model
^^^^^^^^^^^^^^^^

The next step is to train a neural network on our example stories and NLU data.
To do this, run the command below. This command will call the Rasa Core and NLU train
functions, and store the trained model
into the ``models/`` directory. The output of this command will include
the training results for each training epoch.

.. runnable::
   :description: stack-train

   rasa train

   print("Finished training! You can move on to the next step!")

The ``rasa train`` command will look for both NLU and Core data and will train a model for each.
You could have also typed ``rasa train core`` to be explicit, but since the NLU data has not changed
Rasa will automatically skip training an NLU model.


8. Talk To Your Assistant
^^^^^^^^^^^^^^^^^^^^^^^^^

Congratulations 🚀! You just built an assistant
powered entirely by machine learning.

The easiest way to try it out and start collecting new training
data is to use Rasa X.

.. button::
   :text: Talk to Your Assistant
   :link: ../rasax/build


You can also talk to your assistant on the command line without using Rasa X.
To do this, run:

.. runnable::
   :description: stack-chat-with-bot

   rasa shell


.. raw:: html
   :file: poll.html

