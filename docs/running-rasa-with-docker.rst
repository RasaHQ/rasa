:desc: Run and ship your Rasa assistant with Docker containers as on any docker
       compatible machine or cluster.

.. _docker_guide:

Running Rasa with Docker
========================

This is a guide on how to set up Rasa using Docker.
If you have not used Rasa before it is recommended to read the :ref:`project-structure`.

.. contents::

1. Installing Docker
--------------------

If you are not sure whether Docker is installed on your machine execute the
following command:

  .. code-block:: bash

    docker -v && docker-compose -v
    # Docker version 18.09.2, build 6247962
    # docker-compose version 1.23.2, build 1110ad01

If Docker is installed on your machine, the command above will print the
versions of docker and docker-compose. If not -- please install Docker.
See `this instruction page <https://docs.docker.com/install/>`_ for the
instructions.

2. Creating a Chatbot Using Rasa
-------------------------------------

This section will cover the following:

    - Setup of simple chatbot
    - Training of the Rasa model using Docker
    - Running the chatbot using Docker

2.1 Setup
~~~~~~~~~

Start by creating a directory ``data`` in your project directory.
Add a file ``nlu.md`` to your ``data`` directory which includes the training data
for the natural language understanding:

.. code-block:: bash

  mkdir data
  touch data/nlu.md

Then add some examples to each intent, e.g.:

.. code-block:: md

  ## intent:greet
  - hey
  - hello
  - hi
  - good morning
  - good evening
  - hey there

  ## intent:mood_happy
  - perfect
  - very good
  - great
  - amazing
  - wonderful
  - I am feeling very good
  - I am great
  - I'm good

  ## intent:mood_unhappy
  - sad
  - very sad
  - unhappy
  - bad
  - very bad
  - awful
  - terrible
  - not very good
  - extremely sad
  - so sad

  ## intent:goodbye
  - bye
  - goodbye
  - see you around
  - see you later


Then create a file called ``stories.md`` in this directory which will contain the
stories to train your chatbot:

.. code-block:: bash

  touch data/stories.md

Next add some stories to ``data/stories.md``, e.g.:

.. code-block:: md

  ## happy_path
  * greet
    - utter_greet
  * mood_happy
    - utter_happy
  * goodbye
    - utter_goodbye

  ## sad_path
  * greet
    - utter_greet
  * mood_unhappy
    - utter_cheer_up
  * goodbye
    - utter_goodbye

After defining the training data for your chatbot, you have to define its domain.
To do so create a file ``domain.yml`` in your project directory:

.. code-block:: bash

  touch domain.yml

Then add the user intents, the actions of your chatbot, and the templates
for the chatbot responses to ``domain.yml``:

.. code-block:: yaml

    intents:
      - greet
      - mood_happy
      - mood_unhappy
      - goodbye

    actions:
      - utter_greet
      - utter_happy
      - utter_cheer_up
      - utter_goodbye

    templates:
      utter_greet:
        - text: "Hi, how is it going?"
      utter_happy:
        - text: "Great, carry on!"
      utter_cheer_up:
        - text: "Don't be sad. Keep smiling!"
      utter_goodbye:
        - text: "Goodbye!"

.. _model_training_docker:

2.2 Training the Rasa Model
~~~~~~~~~~~~~~~~~~~~~~~~~~~


Now you can train the Rasa model using the following command:

.. code-block:: bash

  docker run \
    -v $(pwd):/app \
    rasa/rasa:latest-full \
    train \
      --domain project/domain.yml \
      --stories project/data/stories.md \
      --out models

Command Description:

  - ``-v $(pwd):/app``: Mounts your project directory into the Docker
    container so that Rasa can train a model on your training data
  - ``rasa/rasa:latest-full``: Use the Rasa image with the tag ``latest-full``
  - ``train``: Execute the ``rasa train`` command within the container. This requires
    the default locations for the configuration files and training data. For more
    information see :ref:`cli-usage`.


This created a directory called ``models`` which contains the trained Rasa model.

2.3 Speaking to the AI Assistant
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You can now test the trained model.

.. code-block:: bash

  docker run \
    -it \
    -v $(pwd)/models:/app/models \
    rasa/rasa:latest-full \
    shell

Command Description:

  - ``-it``: Runs the Docker container in interactive mode so that you can
    interact with the console of the container
  - ``-v $(pwd)/models:/app/models``: Mounts the directory with the trained Rasa model
    in the container
  - ``rasa/rasa:latest-full``: Use the Rasa image with the tag ``latest-full``
  - ``shell``: Executes the ``rasa shell`` command which connects to the chatbot on the
    command line. For more information see :ref:`cli-usage`.

.. _running_docker_container:

2.4 Running the AI Assistant
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To run your AI assistant in production, configure your required
:ref:`messaging-and-voice-channels` in ``credentials.yml``. If the files does not
exist, create it using:

.. code-block:: bash

  touch credentials.yml

Then edit it according to your connected channels.
After this run the trained model with:

.. code-block:: bash

  docker run \
    -v $(pwd)/models:/app/models \
    rasa/rasa:latest-full \
    run

Command Description:

  - ``-v $(pwd)/models:/app/models``: Mounts the directory with the trained Rasa model
    in the container
  - ``rasa/rasa:latest-full``: Use the Rasa image with the tag ``latest-full``
  - ``run``: Executes the ``rasa run`` command. For more information see
    :ref:`cli-usage`.

2.5 Using a Custom Policy
~~~~~~~~~~~~~~~~~~~~~~~~~

If you have a custom policy configuration, you can set that in the ``config
.yml``. If this file currently does not exist in your project directory, create it:

.. code-block:: bash

  touch config.yml

Put your policy configuration in there, e.g.:

.. code-block:: yaml

  policies:
  - name: MemoizationPolicy
  - name: KerasPolicy

Then make sure ``config.yml`` is mounted as file or through its parent directory (if
you are following the guide it is mounted through the project directory).

2.6 Using a Custom NLU Pipeline
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you want to configure the components of your :ref:`choosing_pipeline` you can
configure that in the ``config.yml`` file. If this file currently does not exist in
your project directory, create it:

.. code-block:: bash

  touch config.yml


Put the description of your custom pipeline in there, e.g.:

.. code-block:: yaml

  pipeline:
  - name: "SpacyNLP"
  - name: "SpacyTokenizer"
  - name: "RegexFeaturizer"
  - name: "SpacyFeaturizer"
  - name: "CRFEntityExtractor"
  - name: "SklearnIntentClassifier"

Then retrain your model as described earlier in :ref:`model_training_docker`.

Depending on the selected
`NLU Pipeline <https://rasa.com/docs/nlu/choosing_pipeline/>`_ you might
have to use a different Rasa NLU image:

  - ``rasa/rasa_nlu:latest-spacy``: To use the ``pretrained_embeddings_spacy`` pipeline
  - ``rasa/rasa_nlu:latest-tensorflow``: To use the ``supervised_embeddings``
    pipeline
  - ``rasa/rasa_nlu:latest-mitie``: To use a pipeline which includes ``mitie``
  - ``rasa/rasa_nlu:latest-full``: To build a pipeline with dependencies to
    spaCy and TensorFlow
  - ``rasa/rasa_nlu:latest-bare``: To start with minimal dependencies so
    that you can then add your own

Then make sure ``config.yml`` is mounted as file or through its parent directory (if
you are following the guide it is mounted through the project directory).

.. note::

    If you are using a custom NLU component, you have to add the module file to your
    Docker container. Either do this by mounting the file or by including it in the
    image. Make sure that your module is in the Python module search path, e.g. by
    setting the environment variable ``PYTHONPATH=$PYTHONPATH:<directory of your
    module>``.


3. docker-compose Setup
-----------------------

To run Rasa together with other services, such as a server for custom actions, it is
recommend to use `docker compose <https://docs.docker.com/compose/>`_.
*docker-compose* provides an easy way to run multiple containers together without
having to run multiple commands.

Start by creating a file called ``docker-compose.yml``:

.. code-block:: bash

  touch docker-compose.yml

The file starts with the version of the Docker Compose specification that you
want to use, e.g.:

.. code-block:: yaml

  version: '3.0'

Each container is declared as a ``service`` within the docker compose file.
The first service is the ``rasa`` service.

.. code-block:: yaml

  services:
    rasa:
      image: rasa/rasa:latest-full
      ports:
        - 5005:5005
      volumes:
        - ./:/app
      command:
        - run


The command is similar to the ``docker run`` command in :ref:`running_docker_container`.
The ``ports`` part defines a port mapping between the container and your host
system. In this case it makes ``5005`` of the ``rasa`` service available on
port ``5005`` of your host.
This is the port of the :ref:`rest_channels` interface of Rasa.

.. note::

    Since Docker Compose starts a set of Docker containers it is not longer
    possible to directly connect to one single container after executing the
    ``run`` command.

To run the services configured in your ``docker-compose.yml`` execute:

.. code-block:: bash

    docker-compose up


4. Adding Custom Actions
------------------------

To create more sophisticated chatbots you will probably use :ref:`custom-actions`.
Continuing the example from above you might want to add an action which tells
the user a joke to cheer the user up.

4.1 Creating a Custom Action
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Start with creating the custom actions in a directory ``actions``:

.. code-block:: bash

  mkdir actions
  # Rasa Core SDK expects a python module.
  # Therefore, make sure that you have this file in the directory.
  touch actions/__init__.py
  touch actions/actions.py

Then build a custom action using the Rasa Core SDK, e.g.:

.. code-block:: python

  import requests
  import json
  from rasa_core_sdk import Action


  class ActionJoke(Action):
    def name(self):
      return "action_joke"

    def run(self, dispatcher, tracker, domain):
      request = requests.get('http://api.icndb.com/jokes/random').json() #make an api call
      joke = request['value']['joke'] #extract a joke from returned json response
      dispatcher.utter_message(joke) #send the message back to the user
      return []

Next add the custom action in your stories and your domain file.
Continuing the example from above replace ``utter_cheer_up`` in
``data/stories.md`` with the custom action ``action_joke`` and add
``action_joke`` to the actions in the domain file.

4.2 Adding the Action Server
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The custom actions are run by the action server.
To spin it up together with the Rasa instance, add a service
``action_server`` to the ``docker-compose.yml``:

.. code-block:: yaml

  action_server:
    image: rasa/rasa_core_sdk:latest
    volumes:
      - ./actions:/app/actions

This pulls the image for the Rasa Core SDK which includes the action server,
mounts your custom actions into it, and starts the server.

To instruct Rasa to use the action server you have to tell Rasa its location.
Add this to your ``endpoints.yml`` (if it does not exist, create it):

.. code-block:: yaml

  action_endpoint:
    url: http://action_server:5055/webhook

Run ``docker-compose up`` to start the action server together
with Rasa.

4.3 Adding Custom Dependencies
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If your custom action has additional dependencies, either systems or python libraries,
you can add these by extending the official image.

To do so create a Dockerfile, extend the official image and add your custom
dependencies, e.g.:

.. code-block:: docker

    # Extend the official Rasa Core SDK image
    FROM rasa/rasa_core_sdk:latest

    # Add a custom system library (e.g. git)
    RUN apt-get update && \
        apt-get install -y git

    # Add a custom python library (e.g. jupyter)
    RUN pip install --no-cache-dir \
        jupyter

You can then build the image and use it in your ``docker-compose.yml``:

.. code-block:: bash

  docker build . -t <name of your custom image>:<tag of your custom image>

5. Adding a Custom Tracker Store
--------------------------------

By default all conversations are saved in-memory. This mean that all
conversations are lost as soon as you restart Rasa.
If you want to persist your conversations, you can use different
:ref:`tracker_store`.

5.1 Using MongoDB as Tracker Store
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Start by adding MongoDB to your docker-compose file. The following example
adds the MongoDB as well as a UI (you can skip this), which will be available
at ``localhost:8081``. Username and password for the MongoDB instance are
specified as ``rasa`` and ``example``. For example:

.. code-block:: yaml

  mongo:
    image: mongo
    environment:
      MONGO_INITDB_ROOT_USERNAME: rasa
      MONGO_INITDB_ROOT_PASSWORD: example
  mongo-express:
    image: mongo-express
    ports:
      - 8081:8081
    environment:
      ME_CONFIG_MONGODB_ADMINUSERNAME: rasa
      ME_CONFIG_MONGODB_ADMINPASSWORD: example

Then add the MongoDB to the ``tracker_store`` section of your endpoints
configuration ``config/endpoints.yml``:

.. code-block:: yaml

  tracker_store:
    type: mongod
    url: mongodb://mongo:27017
    username: rasa
    password: example

Then start all components with ``docker-compose up``.

5.2 Using Redis as Tracker Store
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Start by adding Redis to your docker-compose file:

.. code-block:: yaml

  redis:
    image: redis:latest

Then add Redis to the ``tracker_store`` section of your endpoint
configuration ``config/endpoints.yml``:

.. code-block:: yaml

  tracker_store:
    type: redis
    url: redis

5.3 Using a Custom Tracker Store Implementation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you have a custom implementation of a tracker store you have two options
to add this store to Rasa:

  - extending the Rasa image
  - mounting it as volume

Then add the required configuration to your endpoint configuration
``endpoints.yml`` as it is described in :ref:`tracker_store`.
If you want the tracker store component (e.g. a certain database) to be part
of your docker compose file, add a corresponding service and configuration
there.

.. include:: feedback.inc