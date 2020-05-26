:desc: Command line interface for open source chatbot framework Rasa.  Learn how to train, test and run your machine learning-based conversational AI assistants a
 a
.. _command-line-interface: a
 a
Command Line Interface a
====================== a
 a
.. edit-link:: a
 a
 a
.. contents:: a
   :local: a
 a
Cheat Sheet a
~~~~~~~~~~~ a
 a
The command line interface (CLI) gives you easy-to-remember commands for common tasks. a
 a
=========================  ============================================================================================= a
Command                    Effect a
=========================  ============================================================================================= a
``rasa init``              Creates a new project with example training data, actions, and config files. a
``rasa train``             Trains a model using your NLU data and stories, saves trained model in ``./models``. a
``rasa interactive``       Starts an interactive learning session to create new training data by chatting. a
``rasa shell``             Loads your trained model and lets you talk to your assistant on the command line. a
``rasa run``               Starts a Rasa server with your trained model. See the :ref:`configuring-http-api` docs for details. a
``rasa run actions``       Starts an action server using the Rasa SDK. a
``rasa visualize``         Visualizes stories. a
``rasa test``              Tests a trained Rasa model using your test NLU data and stories. a
``rasa data split nlu``    Performs a split of your NLU data according to the specified percentages. a
``rasa data convert nlu``  Converts NLU training data between different formats. a
``rasa export``            Export conversations from a tracker store to an event broker. a
``rasa x``                 Launch Rasa X locally. a
``rasa -h``                Shows all available commands. a
=========================  ============================================================================================= a
 a
 a
Create a new project a
~~~~~~~~~~~~~~~~~~~~ a
 a
A single command sets up a complete project for you with some example training data. a
 a
.. code:: bash a
 a
   rasa init a
 a
 a
This creates the following files: a
 a
.. code:: bash a
 a
   . a
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
 a
The ``rasa init`` command will ask you if you want to train an initial model using this data. a
If you answer no, the ``models`` directory will be empty. a
 a
With this project setup, common commands are very easy to remember. a
To train a model, type ``rasa train``, to talk to your model on the command line, ``rasa shell``, a
to test your model type ``rasa test``. a
 a
 a
Train a Model a
~~~~~~~~~~~~~ a
 a
The main command is: a
 a
.. code:: bash a
 a
   rasa train a
 a
 a
This command trains a Rasa model that combines a Rasa NLU and a Rasa Core model. a
If you only want to train an NLU or a Core model, you can run ``rasa train nlu`` or ``rasa train core``. a
However, Rasa will automatically skip training Core or NLU if the training data and config haven't changed. a
 a
``rasa train`` will store the trained model in the directory defined by ``--out``. The name of the model a
is per default ``<timestamp>.tar.gz``. If you want to name your model differently, you can specify the name a
using ``--fixed-model-name``. a
 a
The following arguments can be used to configure the training process: a
 a
.. program-output:: rasa train --help a
 a
 a
.. note:: a
 a
    Make sure training data for Core and NLU are present when training a model using ``rasa train``. a
    If training data for only one model type is present, the command automatically falls back to a
    ``rasa train nlu`` or ``rasa train core`` depending on the provided training files. a
 a
 a
Interactive Learning a
~~~~~~~~~~~~~~~~~~~~ a
 a
To start an interactive learning session with your assistant, run a
 a
.. code:: bash a
 a
   rasa interactive a
 a
 a
If you provide a trained model using the ``--model`` argument, the interactive learning process a
is started with the provided model. If no model is specified, ``rasa interactive`` will a
train a new Rasa model with the data located in ``data/`` if no other directory was passed to the a
``--data`` flag. After training the initial model, the interactive learning session starts. a
Training will be skipped if the training data and config haven't changed. a
 a
The full list of arguments that can be set for ``rasa interactive`` is: a
 a
.. program-output:: rasa interactive --help a
 a
Talk to your Assistant a
~~~~~~~~~~~~~~~~~~~~~~ a
 a
To start a chat session with your assistant on the command line, run: a
 a
.. code:: bash a
 a
   rasa shell a
 a
The model that should be used to interact with your bot can be specified by ``--model``. a
If you start the shell with an NLU-only model, ``rasa shell`` allows a
you to obtain the intent and entities of any text you type on the command line. a
If your model includes a trained Core model, you can chat with your bot and see a
what the bot predicts as a next action. a
If you have trained a combined Rasa model but nevertheless want to see what your model a
extracts as intents and entities from text, you can use the command ``rasa shell nlu``. a
 a
To increase the logging level for debugging, run: a
 a
.. code:: bash a
 a
   rasa shell --debug a
 a
.. note:: a
   In order to see the typical greetings and/or session start behavior you might see a
   in an external channel, you will need to explicitly send ``/session_start`` a
   as the first message. Otherwise, the session start behavior will begin as described in a
   :ref:`session_config`. a
 a
The full list of options for ``rasa shell`` is: a
 a
.. program-output:: rasa shell --help a
 a
 a
Start a Server a
~~~~~~~~~~~~~~ a
 a
To start a server running your Rasa model, run: a
 a
.. code:: bash a
 a
   rasa run a
 a
The following arguments can be used to configure your Rasa server: a
 a
.. program-output:: rasa run --help a
 a
For more information on the additional parameters, see :ref:`configuring-http-api`. a
See the Rasa :ref:`http-api` docs for detailed documentation of all the endpoints. a
 a
.. _run-action-server: a
 a
Start an Action Server a
~~~~~~~~~~~~~~~~~~~~~~ a
 a
To run your action server run a
 a
.. code:: bash a
 a
   rasa run actions a
 a
The following arguments can be used to adapt the server settings: a
 a
.. program-output:: rasa run actions --help a
 a
 a
Visualize your Stories a
~~~~~~~~~~~~~~~~~~~~~~ a
 a
To open a browser tab with a graph showing your stories: a
 a
.. code:: bash a
 a
   rasa visualize a
 a
Normally, training stories in the directory ``data`` are visualized. If your stories are located a
somewhere else, you can specify their location with ``--stories``. a
 a
Additional arguments are: a
 a
.. program-output:: rasa visualize --help a
 a
 a
Evaluating a Model on Test Data a
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ a
 a
To evaluate your model on test data, run: a
 a
.. code:: bash a
 a
   rasa test a
 a
 a
Specify the model to test using ``--model``. a
Check out more details in :ref:`nlu-evaluation` and :ref:`core-evaluation`. a
 a
The following arguments are available for ``rasa test``: a
 a
.. program-output:: rasa test --help a
 a
 a
.. _train-test-split: a
 a
Create a Train-Test Split a
~~~~~~~~~~~~~~~~~~~~~~~~~ a
 a
To create a split of your NLU data, run: a
 a
.. code:: bash a
 a
   rasa data split nlu a
 a
 a
You can specify the training data, the fraction, and the output directory using the following arguments: a
 a
.. program-output:: rasa data split nlu --help a
 a
 a
This command will attempt to keep the proportions of intents the same in train and test. a
If you have NLG data for retrieval actions, this will be saved to seperate files: a
 a
.. code-block:: bash a
 a
   ls train_test_split a
 a
         nlg_test_data.md     test_data.json a
         nlg_training_data.md training_data.json a
 a
Convert Data Between Markdown and JSON a
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ a
 a
To convert NLU data from LUIS data format, WIT data format, Dialogflow data format, JSON, or Markdown a
to JSON or Markdown, run: a
 a
.. code:: bash a
 a
   rasa data convert nlu a
 a
You can specify the input file, output file, and the output format with the following arguments: a
 a
.. program-output:: rasa data convert nlu --help a
 a
 a
.. _section_export: a
 a
Export Conversations to an Event Broker a
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ a
 a
To export events from a tracker store using an event broker, run: a
 a
.. code:: bash a
 a
   rasa export a
 a
You can specify the location of the environments file, the minimum and maximum a
timestamps of events that should be published, as well as the conversation IDs that a
should be published. a
 a
.. program-output:: rasa export --help a
 a
 a
.. _section_evaluation: a
 a
Start Rasa X a
~~~~~~~~~~~~ a
 a
.. raw:: html a
 a
    Rasa X is a toolset that helps you leverage conversations to improve your assistant. a
    You can find more information about it <a class="reference external" href="https://rasa.com/docs/rasa-x/" target="_blank">here</a>. a
 a
You can start Rasa X locally by executing a
 a
.. code:: bash a
 a
   rasa x a
 a
.. raw:: html a
 a
    To be able to start Rasa X you need to have Rasa X local mode installed a
    and you need to be in a Rasa project. a
 a
.. note:: a
 a
    By default Rasa X runs on the port 5002. Using the argument ``--rasa-x-port`` allows you to change it to a
    any other port. a
 a
The following arguments are available for ``rasa x``: a
 a
.. program-output:: rasa x --help a
 a