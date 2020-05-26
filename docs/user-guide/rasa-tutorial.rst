:desc: This tutorial will show you the different parts needed to build a a
       chatbot or AI assistant using open source Rasa. a
 a
.. _rasa-tutorial: a
 a
Tutorial: Rasa Basics a
===================== a
 a
.. edit-link:: a
 a
This page explains the basics of building an assistant with Rasa and a
shows the structure of a Rasa project. You can test it out right here without a
installing anything. a
You can also :ref:`install Rasa <installation>` and follow along in your command line. a
 a
.. raw:: html a
 a
    The <a style="text-decoration: none" href="https://rasa.com/docs/rasa/glossary">glossary</a> contains an overview of the most common terms you’ll see in the Rasa documentation. a
 a
 a
 a
.. contents:: a
   :local: a
 a
 a
In this tutorial, you will build a simple, friendly assistant which will ask how you're doing a
and send you a fun picture to cheer you up if you are sad. a
 a
.. image:: /_static/images/mood_bot.png a
 a
 a
1. Create a New Project a
^^^^^^^^^^^^^^^^^^^^^^^ a
 a
The first step is to create a new Rasa project. To do this, run: a
 a
 a
 a
.. runnable:: a
 a
   rasa init --no-prompt a
 a
 a
The ``rasa init`` command creates all the files that a Rasa project needs and a
trains a simple bot on some sample data. a
If you leave out the ``--no-prompt`` flag you will be asked some questions about a
how you want your project to be set up. a
 a
This creates the following files: a
 a
 a
+-------------------------------+--------------------------------------------------------+ a
| ``__init__.py``               | an empty file that helps python find your actions      | a
+-------------------------------+--------------------------------------------------------+ a
| ``actions.py``                | code for your custom actions                           | a
+-------------------------------+--------------------------------------------------------+ a
| ``config.yml`` '*'            | configuration of your NLU and Core models              | a
+-------------------------------+--------------------------------------------------------+ a
| ``credentials.yml``           | details for connecting to other services               | a
+-------------------------------+--------------------------------------------------------+ a
| ``data/nlu.md`` '*'           | your NLU training data                                 | a
+-------------------------------+--------------------------------------------------------+ a
| ``data/stories.md`` '*'       | your stories                                           | a
+-------------------------------+--------------------------------------------------------+ a
| ``domain.yml`` '*'            | your assistant's domain                                | a
+-------------------------------+--------------------------------------------------------+ a
| ``endpoints.yml``             | details for connecting to channels like fb messenger   | a
+-------------------------------+--------------------------------------------------------+ a
| ``models/<timestamp>.tar.gz`` | your initial model                                     | a
+-------------------------------+--------------------------------------------------------+ a
 a
 a
 a
The most important files are marked with a '*'. a
You will learn about all of these in this tutorial. a
 a
 a
2. View Your NLU Training Data a
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ a
 a
The first piece of a Rasa assistant is an NLU model. a
NLU stands for Natural Language Understanding, which means turning a
user messages into structured data. To do this with Rasa, a
you provide training examples that show how Rasa should understand a
user messages, and then train a model by showing it those examples. a
 a
Run the code cell below to see the NLU training data created by a
the ``rasa init`` command: a
 a
 a
.. runnable:: a
 a
   cat data/nlu.md a
 a
 a
 a
 a
The lines starting with ``##`` define the names of your ``intents``, which a
are groups of messages with the same meaning. Rasa's job will be to a
predict the correct intent when your users send new, unseen messages to a
your assistant. You can find all the details of the data format in :ref:`training-data-format`. a
 a
.. _model-configuration: a
 a
3. Define Your Model Configuration a
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ a
 a
The configuration file defines the NLU and Core components that your model a
will use. In this example, your NLU model will use the a
``supervised_embeddings`` pipeline. You can learn about the different NLU pipelines a
:ref:`here <choosing-a-pipeline>`. a
 a
Let's take a look at your model configuration file. a
 a
.. runnable:: a
 a
   cat config.yml a
 a
 a
 a
The ``language`` and ``pipeline`` keys specify how the NLU model should be built. a
The ``policies`` key defines the :ref:`policies <policies>` that the Core model will use. a
 a
 a
 a
4. Write Your First Stories a
^^^^^^^^^^^^^^^^^^^^^^^^^^^ a
 a
At this stage, you will teach your assistant how to respond to your messages. a
This is called dialogue management, and is handled by your Core model. a
 a
Core models learn from real conversational data in the form of training "stories". a
A story is a real conversation between a user and an assistant. a
Lines with intents and entities reflect the user's input and action names show what the a
assistant should do in response. a
 a
Below is an example of a simple conversation. a
The user says hello, and the assistant says hello back. a
This is how it looks as a story: a
 a
.. code-block:: story a
 a
   ## story1 a
   * greet a
      - utter_greet a
 a
 a
You can see the full details in :ref:`stories`. a
 a
Lines that start with ``-`` are actions taken by the assistant. a
In this tutorial, all of our actions are messages sent back to the user, a
like ``utter_greet``, but in general, an action can do anything, a
including calling an API and interacting with the outside world. a
 a
Run the command below to view the example stories inside the file ``data/stories.md``: a
 a
 a
.. runnable:: a
 a
   cat data/stories.md a
 a
 a
 a
5. Define a Domain a
^^^^^^^^^^^^^^^^^^ a
 a
The next thing we need to do is define a :ref:`Domain <domains>`. a
The domain defines the universe your assistant lives in: what user inputs it a
should expect to get, what actions it should be able to predict, how to a
respond, and what information to store. a
The domain for our assistant is saved in a a
file called ``domain.yml``: a
 a
 a
 a
.. runnable:: a
 a
   cat domain.yml a
 a
 a
 a
So what do the different parts mean? a
 a
 a
+---------------+-------------------------------------------------------------+ a
| ``intents``   | things you expect users to say                              | a
+---------------+-------------------------------------------------------------+ a
| ``actions``   | things your assistant can do and say                        | a
+---------------+-------------------------------------------------------------+ a
| ``responses`` | responses for the things your assistant can say             | a
+---------------+-------------------------------------------------------------+ a
 a
 a
**How does this fit together?** a
Rasa Core's job is to choose the right action to execute at each step a
of the conversation. In this case, our actions simply send a message to the user. a
These simple utterance actions are the ``actions`` in the domain that start a
with ``utter_``. The assistant will respond with a message based on a response a
from the ``responses`` section. See :ref:`custom-actions` a
to build actions that do more than just send a message. a
 a
 a
 a
6. Train a Model a
^^^^^^^^^^^^^^^^ a
 a
Anytime we add new NLU or Core data, or update the domain or configuration, we a
need to re-train a neural network on our example stories and NLU data. a
To do this, run the command below. This command will call the Rasa Core and NLU train a
functions and store the trained model a
into the ``models/`` directory. The command will automatically only retrain the a
different model parts if something has changed in their data or configuration. a
 a
 a
 a
.. runnable:: a
 a
   rasa train a
 a
   echo "Finished training." a
 a
 a
 a
The ``rasa train`` command will look for both NLU and Core data and will train a combined model. a
 a
7. Test Your Assistant a
^^^^^^^^^^^^^^^^^^^^^^ a
 a
After you train a model, you always want to check that your assistant still behaves as you expect. a
In Rasa Open Source, you use end-to-end tests defined in your ``tests/`` directory to run through a
test conversations that ensure both NLU and Core make correct predictions. a
 a
.. runnable:: a
 a
   rasa test a
 a
   echo "Finished running tests." a
 a
See :ref:`testing-your-assistant` to learn more about how to evaluate your model as you improve it. a
 a
8. Talk to Your Assistant a
^^^^^^^^^^^^^^^^^^^^^^^^^ a
 a
Congratulations! 🚀 You just built an assistant a
powered entirely by machine learning. a
 a
The next step is to try it out! a
If you're following this tutorial on your local machine, start talking to your a
assistant by running: a
 a
.. code-block:: bash a
 a
   rasa shell a
 a
 a
Next Steps a
^^^^^^^^^^ a
 a
Now that you've built your first Rasa bot it's time to learn about a
some more advanced Rasa features. a
 a
- Learn how to implement business logic using :ref:`forms <forms>` a
- Learn how to integrate other APIs using :ref:`custom actions <actions>` a
- Learn how to connect your bot to different :ref:`messaging apps <messaging-and-voice-channels>` a
- Learn about customising the :ref:`components <components>` in your NLU pipeline a
- Read about custom and built-in :ref:`entities <entity-extraction>` a
 a
You can also use Rasa X to collect more conversations a
and improve your assistant: a
 a
.. button:: a
   :text: Try Rasa X a
   :link: ../../../rasa-x/ a
 a
.. juniper:: a
   :language: bash a
 a