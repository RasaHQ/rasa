.. _command-line-interface:

Command Line Interface
======================


.. contents::
   :local:

Cheat Sheet
~~~~~~~~~~~

The command line interface (CLI) gives you easy-to-remember commands for common tasks.

=========================  =============================================================================================
Command                    Effect
=========================  =============================================================================================
``rasa init``              Creates a new project with example training data, actions, and config files.
``rasa train``             Trains a model using your NLU data and stories, saves trained model in ``./models``.
``rasa interactive``       Starts an interactive learning session to create new training data by chatting.
``rasa shell``             Loads your trained model and lets you talk to your assistant on the command line.
``rasa run``               Starts a Rasa server with your trained model. See the :ref:`running-the-server` docs for details.
``rasa run actions``       Starts an action server using the Rasa SDK.
``rasa visualize``         Visualizes stories.
``rasa test``              Tests a trained Rasa model using your test NLU data and stories.
``rasa data split nlu``    Performs a split of your NLU data according to the specified percentages.
``rasa data convert nlu``  Converts NLU training data between different formats.
``rasa -h``                Shows all available commands.
=========================  =============================================================================================


Create a new project
~~~~~~~~~~~~~~~~~~~~

A single command sets up a complete project for you with some example training data.

.. code:: bash

   rasa init


This creates the following files:

.. code:: bash

   .
   ├── __init__.py
   ├── actions.py
   ├── config.yml
   ├── credentials.yml
   ├── data
   │   ├── nlu.md
   │   └── stories.md
   ├── domain.yml
   ├── endpoints.yml
   └── models
       └── <timestamp>.tar.gz

The ``rasa init`` command will ask you if you want to train an initial model using this data.
If you answer no, the ``models`` directory will be empty.

With this project setup, common commands are very easy to remember.
To train a model, type ``rasa train``, to talk to your model on the command line, ``rasa shell``,
to test your model type ``rasa test``.


Train a Model
~~~~~~~~~~~~~

The main command is:

.. code:: bash

   rasa train


This command trains a Rasa model that combines a Rasa NLU and a Rasa Core model.
If you only want to train an NLU or a Core model, you can run ``rasa train nlu`` or ``rasa train core``.
However, Rasa will automatically skip training Core or NLU if the training data and config haven't changed.

``rasa train`` will store the trained model in the directory defined by ``--out``. The name of the model
is per default ``<timestamp>.tar.gz``. If you want to name your model differently, you can specify the name
using ``--fixed-model-name``.

The following arguments can be used to configure the training process:

.. program-output:: rasa train --help


.. note::

    Make sure training data for Core and NLU are present when training a model using ``rasa train``.
    If training data for only one model type is present, the command automatically falls back to
    ``rasa train nlu`` or ``rasa train core`` depending on the provided training files.


Interactive Learning
~~~~~~~~~~~~~~~~~~~~

To start an interactive learning session with your assistant, run

.. code:: bash

   rasa interactive


If you provide a trained model using the ``--model`` argument, the interactive learning process
is started with the provided model. If no model is specified, ``rasa interactive`` will
train a new Rasa model with the data located in ``data/`` if no other directory was passed to the
``--data`` flag. After training the initial model, the interactive learning session starts.
Training will be skipped if the training data and config haven't changed.

The full list of arguments that can be set for ``rasa interactive`` is:

.. program-output:: rasa interactive --help

Talk to your Assistant
~~~~~~~~~~~~~~~~~~~~~~

To start a chat session with your assistant on the command line, run:

.. code:: bash

   rasa shell

The model that should be used to interact with your bot can be specified by ``--model``.
If you start the shell with an NLU-only model, ``rasa shell`` allows
you to obtain the intent and entities of any text you type on the command line.
If your model includes a trained Core model, you can chat with your bot and see
what the bot predicts as a next action.
If you have trained a combined Rasa model but nevertheless want to see what your model
extracts as intents and entities from text, you can use the command ``rasa shell nlu``.

To increase the logging level for debugging, run:

.. code:: bash

   rasa shell --debug


The full list of options for ``rasa shell`` is

.. program-output:: rasa shell --help


Start a Server
~~~~~~~~~~~~~~

To start a server running your Rasa model, run:

.. code:: bash

   rasa run

The following arguments can be used to configure your Rasa server:

.. program-output:: rasa run --help

For more information on the additional parameters, see :ref:`running-the-server`.
See the Rasa :ref:`http-api` docs for detailed documentation of all the endpoints.

.. _run-action-server:

Start an Action Server
~~~~~~~~~~~~~~~~~~~~~~

To run your action server run

.. code:: bash

   rasa run actions

The following arguments can be used to adapt the server settings:

.. program-output:: rasa run actions --help


Visualize your Stories
~~~~~~~~~~~~~~~~~~~~~~

To open a browser tab with a graph showing your stories:

.. code:: bash

   rasa visualize

Normally, training stories in the directory ``data`` are visualized. If your stories are located
somewhere else, you can specify their location with ``--stories``.

Additional arguments are:

.. program-output:: rasa visualize --help


Evaluate a Model on Test Data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To evaluate your model on test data, run:

.. code:: bash

   rasa test


Specify the model to test using ``--model``.
Check out more details in :ref:`nlu-evaluation` and :ref:`core-evaluation`.

The following arguments are available for ``rasa test``:

.. program-output:: rasa test --help


.. _train-test-split:

Create a Train-Test Split
~~~~~~~~~~~~~~~~~~~~~~~~~

To create a split of your NLU data, run:

.. code:: bash

   rasa data split nlu


You can specify the training data, the fraction, and the output directory using the following arguments:

.. program-output:: rasa data split nlu --help


This command will attempt to keep the proportions of intents the same in train and test.


Convert Data Between Markdown and JSON
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To convert NLU data from LUIS data format, WIT data format, Dialogflow data format, json, or Markdown
to json or Markdown, run:

.. code:: bash

   rasa data convert nlu

You can specify the input file, output file, and the output format with the following arguments:

.. program-output:: rasa data convert nlu --help


.. _section_evaluation:
