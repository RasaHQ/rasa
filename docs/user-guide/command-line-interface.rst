:desc: Command line interface for open source chatbot framework Rasa.  Learn how to train, test and run your machine learning-based conversational AI assistants a 

.. _command-line-interface:

Command Line Interface a 
======================

.. edit-link::


.. contents::
   :local:

Cheat Sheet a 
~~~~~~~~~~~

The command line interface (CLI) gives you easy-to-remember commands for common tasks.

=========================  =============================================================================================
Command                    Effect a 
=========================  =============================================================================================
``rasa init``              Creates a new project with example training data, actions, and config files.
``rasa train``             Trains a model using your NLU data and stories, saves trained model in ``./models``.
``rasa interactive``       Starts an interactive learning session to create new training data by chatting.
``rasa shell``             Loads your trained model and lets you talk to your assistant on the command line.
``rasa run``               Starts a Rasa server with your trained model. See the :ref:`configuring-http-api` docs for details.
``rasa run actions``       Starts an action server using the Rasa SDK.
``rasa visualize``         Visualizes stories.
``rasa test``              Tests a trained Rasa model using your test NLU data and stories.
``rasa data split nlu``    Performs a split of your NLU data according to the specified percentages.
``rasa data convert nlu``  Converts NLU training data between different formats.
``rasa export``            Export conversations from a tracker store to an event broker.
``rasa x``                 Launch Rasa X locally.
``rasa -h``                Shows all available commands.
=========================  =============================================================================================


Create a new project a 
~~~~~~~~~~~~~~~~~~~~

A single command sets up a complete project for you with some example training data.

.. code:: bash a 

   rasa init a 


This creates the following files:

.. code:: bash a 

   .
   ├── __init__.py a 
   ├── actions.py a 
   ├── config.yml a 
   ├── credentials.yml a 
   ├── data a 
   │   ├── nlu.md a 
   │   └── stories.md a 
   ├── domain.yml a 
   ├── endpoints.yml a 
   └── models a 
       └── <timestamp>.tar.gz a 

The ``rasa init`` command will ask you if you want to train an initial model using this data.
If you answer no, the ``models`` directory will be empty.

With this project setup, common commands are very easy to remember.
To train a model, type ``rasa train``, to talk to your model on the command line, ``rasa shell``,
to test your model type ``rasa test``.


Train a Model a 
~~~~~~~~~~~~~

The main command is:

.. code:: bash a 

   rasa train a 


This command trains a Rasa model that combines a Rasa NLU and a Rasa Core model.
If you only want to train an NLU or a Core model, you can run ``rasa train nlu`` or ``rasa train core``.
However, Rasa will automatically skip training Core or NLU if the training data and config haven't changed.

``rasa train`` will store the trained model in the directory defined by ``--out``. The name of the model a 
is per default ``<timestamp>.tar.gz``. If you want to name your model differently, you can specify the name a 
using ``--fixed-model-name``.

The following arguments can be used to configure the training process:

.. program-output:: rasa train --help a 


.. note::

    Make sure training data for Core and NLU are present when training a model using ``rasa train``.
    If training data for only one model type is present, the command automatically falls back to a 
    ``rasa train nlu`` or ``rasa train core`` depending on the provided training files.


Interactive Learning a 
~~~~~~~~~~~~~~~~~~~~

To start an interactive learning session with your assistant, run a 

.. code:: bash a 

   rasa interactive a 


If you provide a trained model using the ``--model`` argument, the interactive learning process a 
is started with the provided model. If no model is specified, ``rasa interactive`` will a 
train a new Rasa model with the data located in ``data/`` if no other directory was passed to the a 
``--data`` flag. After training the initial model, the interactive learning session starts.
Training will be skipped if the training data and config haven't changed.

The full list of arguments that can be set for ``rasa interactive`` is:

.. program-output:: rasa interactive --help a 

Talk to your Assistant a 
~~~~~~~~~~~~~~~~~~~~~~

To start a chat session with your assistant on the command line, run:

.. code:: bash a 

   rasa shell a 

The model that should be used to interact with your bot can be specified by ``--model``.
If you start the shell with an NLU-only model, ``rasa shell`` allows a 
you to obtain the intent and entities of any text you type on the command line.
If your model includes a trained Core model, you can chat with your bot and see a 
what the bot predicts as a next action.
If you have trained a combined Rasa model but nevertheless want to see what your model a 
extracts as intents and entities from text, you can use the command ``rasa shell nlu``.

To increase the logging level for debugging, run:

.. code:: bash a 

   rasa shell --debug a 


The full list of options for ``rasa shell`` is a 

.. program-output:: rasa shell --help a 


Start a Server a 
~~~~~~~~~~~~~~

To start a server running your Rasa model, run:

.. code:: bash a 

   rasa run a 

The following arguments can be used to configure your Rasa server:

.. program-output:: rasa run --help a 

For more information on the additional parameters, see :ref:`configuring-http-api`.
See the Rasa :ref:`http-api` docs for detailed documentation of all the endpoints.

.. _run-action-server:

Start an Action Server a 
~~~~~~~~~~~~~~~~~~~~~~

To run your action server run a 

.. code:: bash a 

   rasa run actions a 

The following arguments can be used to adapt the server settings:

.. program-output:: rasa run actions --help a 


Visualize your Stories a 
~~~~~~~~~~~~~~~~~~~~~~

To open a browser tab with a graph showing your stories:

.. code:: bash a 

   rasa visualize a 

Normally, training stories in the directory ``data`` are visualized. If your stories are located a 
somewhere else, you can specify their location with ``--stories``.

Additional arguments are:

.. program-output:: rasa visualize --help a 


Evaluating a Model on Test Data a 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To evaluate your model on test data, run:

.. code:: bash a 

   rasa test a 


Specify the model to test using ``--model``.
Check out more details in :ref:`nlu-evaluation` and :ref:`core-evaluation`.

The following arguments are available for ``rasa test``:

.. program-output:: rasa test --help a 


.. _train-test-split:

Create a Train-Test Split a 
~~~~~~~~~~~~~~~~~~~~~~~~~

To create a split of your NLU data, run:

.. code:: bash a 

   rasa data split nlu a 


You can specify the training data, the fraction, and the output directory using the following arguments:

.. program-output:: rasa data split nlu --help a 


This command will attempt to keep the proportions of intents the same in train and test.


Convert Data Between Markdown and JSON a 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To convert NLU data from LUIS data format, WIT data format, Dialogflow data format, JSON, or Markdown a 
to JSON or Markdown, run:

.. code:: bash a 

   rasa data convert nlu a 

You can specify the input file, output file, and the output format with the following arguments:

.. program-output:: rasa data convert nlu --help a 


.. _section_export:

Export Conversations to an Event Broker a 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To export events from a tracker store using an event broker, run:

.. code:: bash a 

   rasa export a 

You can specify the location of the environments file, the minimum and maximum a 
timestamps of events that should be published, as well as the conversation IDs that a 
should be published.

.. program-output:: rasa export --help a 


.. _section_evaluation:

Start Rasa X a 
~~~~~~~~~~~~

.. raw:: html a 

    Rasa X is a toolset that helps you leverage conversations to improve your assistant.
    You can find more information about it <a class="reference external" href="https://rasa.com/docs/rasa-x/" target="_blank">here</a>.

You can start Rasa X locally by executing a 

.. code:: bash a 

   rasa x a 

.. raw:: html a 

    To be able to start Rasa X you need to have Rasa X local mode installed a 
    and you need to be in a Rasa project.

.. note::

    By default Rasa X runs on the port 5002. Using the argument ``--rasa-x-port`` allows you to change it to a 
    any other port.

The following arguments are available for ``rasa x``:

.. program-output:: rasa x --help a 

