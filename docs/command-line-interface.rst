.. _cli-usage:

Command Line Interface
======================


.. contents:: 
   :local:

Cheat Sheet
~~~~~~~~~~~

The command line interface (CLI) gives you easy-to-remember commands for common tasks.

=========================  ===================================================================================
Command                    Effect
=========================  ===================================================================================
``rasa init``              Creates a new project, with example training data, actions, and config files.
``rasa run``               Starts a server with your model loaded. See the :ref:`http-api` docs for details.
``rasa run actions``       Starts an action server using the Rasa SDK.
``rasa shell``             Loads your trained model and lets you talk to your assistant on the command line.
``rasa train``             Trains a model using your nlu data and stories, saves trained model in ``./models``.
``rasa interactive``       Starts an interactive learning session, to create new training data by chatting.
``rasa test``              Tests a trained model using your test nlu data and stories.
``rasa show``              Visualize training data.
``rasa data``              Utils for the Rasa training files.
``rasa -h``                Shows all available commands.
=========================  ===================================================================================

.. note::

    You can also see the available options for each subcommand 
    by adding the ``-h`` flag, e.g. ``rasa train -h``


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


Start a Server
~~~~~~~~~~~~~~

To start a server running your Rasa model, run:

.. code:: bash

   rasa run

Among others, the following arguments can be used to configure the server:

.. code:: bash

  -m MODEL, --model MODEL
                        Path to a trained Rasa model. If a directory is
                        specified, it will use the latest model in this
                        directory. (default: models)
  --endpoints ENDPOINTS
                        Configuration file for the model server and the
                        connectors as a yml file. (default: None)
  --enable-api          Start the web server api in addition to the input
                        channel. (default: False)

For more information on the additional parameters, see :ref:`_section_http`.
See the Rasa :ref:`http-api` docs for detailed documentation of all the endpoints.


.. _run-action-server:

Start an Action Server
~~~~~~~~~~~~~~~~~~~~~~

To run your action server run

.. code:: bash

   rasa run actions

The following arguments can be used to adapt the server settings:

.. code:: bash

  -p PORT, --port PORT  port to run the server at (default: 5055)
  --cors [CORS [CORS ...]]
                        enable CORS for the passed origin. Use * to whitelist
                        all origins (default: None)
  --actions ACTIONS     name of action package to be loaded (default: actions)


Talk to your Assistant
~~~~~~~~~~~~~~~~~~~~~~

To start a chat session with your assistant on the command line, run:

.. code:: bash

   rasa shell


The model that should be used to interact with your bot, can be specified by

.. code:: bash

  -m MODEL, --model MODEL
                        Path to a trained Rasa model. If a directory is
                        specified, it will use the latest model in this
                        directory. (default: models)


In case you start the chat session only with a trained NLU model, `rasa shell` allows
you to obtain the intent and entities of any text you type on the command line.
If your model includes a trained Core model, you can chat with your bot and see
what the bot predicts as a next action.

To increase the logging level for debugging, run:

.. code:: bash

   rasa shell --debug


Train a Model
~~~~~~~~~~~~~

The main command is:

.. code:: bash

   rasa train


This command trains a Rasa model that combines a Rasa NLU and a Rasa Core model.
The following arguments allow you to specify the data files, the configuration file, the domain file, and the
output path.

.. code:: bash

  --data DATA [DATA ...]
                        Paths to the Core and NLU training files (default: data).
  -c CONFIG, --config CONFIG
                        The policy and NLU pipeline configuration of your bot (default: config.yml).
  -d DOMAIN, --domain DOMAIN
                        Domain specification (yml file) (default: domain.yml).
  --out OUT             Directory where your models should be stored (default: models).


If you only want to train an NLU or a Core model, you can run ``rasa train nlu`` or ``rasa train core``.
However, Rasa will automatically skip training Core or NLU if the training data and config haven't changed.

.. note::

    Make sure training data for Core and NLU are present when training a model using `rasa train`.
    If only training data for one model type are present, the command automatically falls back to
    `rasa train nlu` or `rasa train core` depending on the provided training files.


Interactive Learning
~~~~~~~~~~~~~~~~~~~~

To start an interactive learning session with your assistant, run

.. code:: bash

   rasa interactive


This command will initially train a Rasa model with the data located in `data`, if no other data directory
was specified. After training the first initial model, the interactive learning session starts. However,
training will be skipped if the training data and config haven't changed.

For training the initial model you can specify the same arguments as for ``rasa train``:

.. code:: bash

  --data DATA [DATA ...]
                        Paths to the Core and NLU training files (default: data).
  -c CONFIG, --config CONFIG
                        The policy and NLU pipeline configuration of your bot (default: config.yml).
  -d DOMAIN, --domain DOMAIN
                        Domain specification (yml file) (default: domain.yml).
  --out OUT             Directory where your models should be stored (default: models).


The interactive learning session starts a Rasa server in the background.
For more information on the additional parameters for the server, see :ref:`_section_http`.


Create a Train-Test Split
~~~~~~~~~~~~~~~~~~~~~~~~~

To create a split of your NLU data, run:

.. code:: bash

   rasa data split nlu


You can specify the training data, the fraction, and the output directory using the following arguments:

.. code:: bash

  -u NLU, --nlu NLU     File or folder containing your NLU training data.
                        (default: data)
  --training-fraction TRAINING_FRACTION
                        Percentage of the data which should be the training
                        data. (default: 0.8)
  --out OUT             Directory where the split files should be stored.
                        (default: train_test_split)


This command will attempt to keep the proportions of intents the same in train and test.


Convert Data Between Markdown and JSON
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To convert nlu data from LUIS data format, WIT data format, json, or markdown to json or markdown, run:

.. code:: bash

   rasa data convert nlu

You can specify the input file, output file, and the output format with the following arguments:

.. code:: bash

  -d DATA_FILE, --data_file DATA_FILE
                        File or directory containing training data. (default:
                        None)
  --out_file OUT_FILE   File where to save training data in Rasa format.
                        (default: None)
  -f {json,md}, --format {json,md}
                        Output format the training data should be converted
                        into. (default: None)


Visualize your Stories
~~~~~~~~~~~~~~~~~~~~~~

To open a browser tab with a graph showing your stories:

.. code:: bash

   rasa show stories

Normally, training stories in the directory ``data`` are visualized. If your training stories are located in a
different location, you can specify the location with

.. code:: bash

  -s STORIES, --stories STORIES
                        File or folder containing training stories. (default:
                        data)

Additional arguments are:

.. code:: bash

  -d DOMAIN, --domain DOMAIN
                        Domain specification (yml file). (default: domain.yml)
  -c CONFIG, --config CONFIG
                        The policy and NLU pipeline configuration of your bot.
                        (default: config.yml)
  --output OUTPUT       Filename of the output path, e.g. 'graph.html'.
                        (default: graph.html)
  --max-history MAX_HISTORY
                        Max history to consider when merging paths in the
                        output graph. (default: 2)
  -nlu NLU_DATA, --nlu-data NLU_DATA
                        Path of the Rasa NLU training data, used to insert
                        example messages into the graph. (default: None)


.. _section_evaluation:

Evaluate a Model on Test Data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To evaluate your model on test data, run:

.. code:: bash

   rasa test

Specify the model to test using:

.. code:: bash

  -m MODEL, --model MODEL
                        Path to a trained Rasa model. If a directory is
                        specified, it will use the latest model in this
                        directory. (default: models)


Check out more details in :ref:`nlu-evaluation` and :ref:`core-evaluation`.
