:desc: Setup open source Rasa Core with Docker in your own infrastructure for on
       premise contextual AI assistants and chatbots. 

.. _docker_walkthrough:

Building Rasa with Docker
=========================

This walkthrough provides a tutorial on how to set up Rasa Core, Rasa NLU,
and an Action Server with Docker containers.
If you have not used Rasa before it is recommended to read the
:ref:`quickstart`.

.. contents::

1. Setup
--------

Requirements for the tutorial:

    - A text editor of your choice
    - Docker

If you are not sure whether Docker is installed on your machine execute the
following command:

  .. code-block:: bash

    docker -v && docker-compose -v
    # Docker version 18.06.1-ce, build e68fc7a
    # docker-compose version 1.22.0, build f46880f

If Docker is installed on your machine, the command above will print the
versions of docker and docker-compose. If not - please install Docker.
See `this instruction page <https://docs.docker.com/install/>`_ for the
instructions.

2. Creating a Chatbot Using Rasa Core
-------------------------------------

This section will cover the following:

    - Setup of simple chatbot
    - Training of the Rasa Core model using Docker
    - Running the chatbot using Docker

2.1 Setup
~~~~~~~~~

Start by creating a directory ``data`` in your project directory. Then create
a file called ``stories.md`` in this directory which will contain the stories
to train your chatbot:

.. code-block:: bash

  mkdir data
  touch data/stories.md

Then add some stories to ``data/stories.md``, e.g.:

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

After defining some training data for your chatbot, you have to define its domain.
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

2.2 Training the Rasa Core Model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Now you can train the Rasa Core model using the following command:

.. code-block:: bash

  docker run \
    -v $(pwd):/app/project \
    -v $(pwd)/models/rasa_core:/app/models \
    rasa/rasa:latest-full \
    train \
      --domain project/domain.yml \
      --stories project/data/stories.md \
      --out models

Command Description:

  - ``-v $(pwd):/app/project``: Mounts your project directory into the Docker
    container so that Rasa Core can train a model on your story data and the
    domain file
  - ``-v $(pwd)/models/rasa_core:/app/models``: Mounts the directory
    `models/rasa_core` in the container which is used to store the
    trained Rasa Core model.
  - ``rasa/rasa_core:latest``: Use the Rasa Core image with the tag ``latest``
  - ``train``: Execute the ``train`` command within the container with

    - ``--domain project/domain.yml``: Path to your domain file from within the
      container
    - ``--stories project/data/stories.md``: Path to your training stories from
      within the container
    - ``--out models``: Instructs Rasa Core to store the trained model in the
      directory ``models`` which corresponds to your host directory
      ``models/rasa_core``

This should have created a directory called ``models/rasa_core`` which contains
the trained Rasa Core model.

2.3 Testing
~~~~~~~~~~~

You can now test the trained model. Keep in mind
that there is currently no Rasa NLU set up. Therefore, you have to explicitly
specify the user intent using the ``/`` prefix, e.g. ``/greet``.
Use the following command to run Rasa Core:

.. code-block:: bash

  docker run \
    -it \
    -v $(pwd)/models/rasa_core:/app/models \
    rasa/rasa:latest-full \
    start \
      --core models

Command Description:

  - ``-it``: Runs the Docker container in interactive mode so that you can
    interact with the console of the container
  - ``-v $(pwd)/models/rasa_core:/app/models``: Mounts the trained Rasa Core
    model in the container
  - ``rasa/rasa_core:latest``: Use the Rasa Core image with the tag ``latest``
  - ``start``: Executes the start command which connects to the chatbot on the
    command line with

    - ``--core models``: Defines the location of the trained model which is
      used for the conversation.


3. Adding Natural Language Understanding (Rasa NLU)
---------------------------------------------------

This section will cover the following:

    - Creation of Rasa NLU training data
    - Training of the Rasa NLU model using Docker
    - Connecting Rasa Core and Rasa NLU
    - Adding a custom NLU pipeline

3.1 Adding NLU Training Data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To add some Rasa NLU training data, add a file ``nlu.md`` to your ``data``
directory:

.. code-block:: bash

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

3.1 Training the NLU Model
~~~~~~~~~~~~~~~~~~~~~~~~~~

You can then train the Rasa NLU model by executing the command below.
As output of the command the directory ``models/`` will contain
the trained Rasa model:

.. code-block:: bash

  docker run \
    -v $(pwd):/app/project \
    -v $(pwd)/models/:/app/models \
    rasa/rasa:latest-spacy-en \
    run \
      python3 -m rasa.train \
      -c config.yml \
      -d project/data/nlu.md \
      -o models \
      --project current

Command Description:

  - ``-v $(pwd):/app/project``: Mounts your project directory into the Docker
    container so that the chatbot can be trained on your NLU data.
  - ``-v $(pwd)/models/:/app/models``: Mounts the directory
    ``models/`` in the container which is used to store the
    trained Rasa model.
  - ``rasa/rasa:latest-spacy``: Using the latest Rasa together with
    the `spaCy` `pipeline <https://rasa.com/docs/nlu/choosing_pipeline/>`_ .
  - ``run``: Entrypoint parameter to run any command within the NLU container
  - ``python3 -m rasa.train``: Starts the NLU training with

    - ``-c config.yml``: Uses the default NLU pipeline configuration which is
      provided by the Docker image
    - ``-d project/data/nlu.md``: Path to the NLU training data
    - ``-o models``: The directory which is used to store the NLU models
    - ``--project current``: The project name to use.

3.2 Connecting Rasa Core and Rasa NLU
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You can connect Rasa Core and Rasa NLU by running each container
individually. However, this setup can get quite complicated
as soon as more components are added. Therefore, it is suggested to use
`docker compose <https://docs.docker.com/compose/>`_ which uses a so called
`compose file` to specify all components and their configuration. This makes it
possible to start all components using a single command.

Start with creating the compose file:

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
        - ./models/:/app/models
      command:
        - start
        - --core
        - models
        - -c
        - rest

The command is similar to the ``docker run`` command in section 2.4.
Note the use of the port mapping and the additional parameters ``-c rest``.
The ``ports`` part defines a port mapping between the container and your host
system. In this case it makes ``5005`` of the ``rasa_core`` service available on
port ``5005`` of your host.
This is the port of the :ref:`rest_channels` interface of Rasa Core.

The parameters ``-c rest`` instruct Rasa Core to use REST as input / output
channel. Since Docker Compose starts a set of Docker containers it is not longer
possible to directly connect to one single container after executing the
``run`` command.

Then add the Rasa NLU service to your docker compose file:

.. code-block:: yaml

  rasa:
      image: rasa/rasa:latest-spacy-en
      volumes:
        - ./models/rasa_nlu:/app/models
      command:
        - start
        - --path
        - models

This maps the Rasa NLU model in the container and instructs Rasa NLU to run
the server for the model.

To instruct Rasa Core to connect to the Rasa NLU server and which NLU model
it should use, it is required to create a file ``config/endpoints.yml`` which
contains the URL Rasa Core should connect to:


.. code-block:: bash

  mkdir config
  touch config/endpoints.yml

Docker containers which are started using Docker Compose are using the same
network. Hence, each service can access other services by their service name.
Therefore, you can use ``rasa_nlu`` as host in ``config/endpoints.yml``:

.. code-block:: yaml

  nlu:
    url: http://rasa_nlu:5000

To make the endpoint configuration available to Rasa Core, you need to mount
the ``config`` directory into the Rasa Core container.
Then instruct Rasa Core to use the endpoints configuration with the parameter
``--endpoints <path to endpoints.yml>`` and define the targeted Rasa NLU model
with ``-u <nlu model to use>``. By adding this additional configuration to
your ``docker-compose.yml`` it should have the following content:

.. code-block:: yaml

  version: '3.0'

  services:
    rasa_core:
      image: rasa/rasa_core:latest
      ports:
        - 5005:5005
      volumes:
        - ./models/rasa_core:/app/models
        - ./config:/app/config
      command:
        - start
        - --core
        - models
        - -c
        - rest
        - --endpoints
        - config/endpoints.yml
        - -u
        - default/
    rasa_nlu:
      image: rasa/rasa_nlu:latest-spacy
      volumes:
        - ./models/rasa_nlu:/app/models
      command:
        - start
        - --path
        - models


3.3 Running Rasa Core and Rasa NLU
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To start Rasa Core and Rasa NLU execute:

.. code-block:: bash

  docker-compose up

.. note::

  Add the flag ``-d`` if you want to run it detached.

The REST API of Rasa Core is then available on ``http://localhost:5005``.
To send messages to your chatbot:

.. code-block:: bash

  curl --request POST \
    --url http://localhost:5005/webhooks/rest/webhook \
    --header 'content-type: application/json' \
    --data '{
      "message": "hello"
    }'

Your chatbot should then answer something like:

.. code-block:: bash

  [
    {
      "recipient_id": "default",
      "text": "Hi, how is it going?"
    }
  ]

If the chatbot cannot understand you, the answer is ``[]``.

3.4 Adding a Custom NLU Pipeline
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you want to configure the components of your
`NLU Pipeline <https://rasa.com/docs/nlu/choosing_pipeline/>`_, start by
creating a file ``nlu_config.yml`` in your ``config`` directory:

.. code-block:: bash

  touch config/nlu_config.yml


Put the description of your custom pipeline in there, e.g.:

.. code-block:: yaml

  pipeline:
  - name: "SpacyNLP"
  - name: "SpacyTokenizer"
  - name: "RegexFeaturizer"
  - name: "SpacyFeaturizer"
  - name: "CRFEntityExtractor"
  - name: "SklearnIntentClassifier"

Then retrain your NLU model. In contrast to the previous training also mount
the ``config`` directory which contains the NLU configuration
and specify it in the run command:

.. code-block:: bash

  docker run \
    -v $(pwd):/app/project \
    -v $(pwd)/models/rasa_nlu:/app/models \
    -v $(pwd)/config:/app/config \
    rasa/rasa_nlu:latest-spacy \
    run \
      python3 -m rasa.train \
      -c config/nlu_config.yml \
      -d project/data/nlu.md \
      -o models \
      --project current

Then adapt the NLU start command in your docker compose so that it uses your
NLU configuration. As in for the training mount the ``config`` directory into
your NLU container and instruct Rasa NLU to use this configuration by adding
the flag ``-c <path to your nlu config>``.
The configuration of the ``rasa_nlu`` server might then look similar to this:

.. code-block:: yaml

  rasa_nlu:
      image: rasa/rasa_nlu:latest-spacy
      volumes:
        - ./models/rasa_nlu:/app/models
        - ./config:/app/config
      command:
        - start
        - --path
        - models
        - -c
        - config/nlu_config.yml

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

4. Adding Custom Actions
------------------------

To create more sophisticated chatbots you will probably use :ref:`customactions`.
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
To spin it up together with Rasa Core and Rasa NLU, add a service
``action_server`` to the ``docker-compose.yml``:

.. code-block:: yaml

  action_server:
    image: rasa/rasa_core_sdk:latest
    volumes:
      - ./actions:/app/actions

This pulls the image for the Rasa Core SDK which includes the action server,
mounts your custom actions into it, and starts the server.

As for Rasa NLU, it is necessary to tell Rasa Core the location of the action
server. Add this to your ``config/endpoints.yml``:

.. code-block:: yaml

  action_endpoint:
    url: http://action_server:5055/webhook

Run ``docker-compose up`` to start the action server together
with Rasa Core and Rasa NLU and to execute your custom actions.

4.3 Adding Custom Dependencies
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If your action has additional dependencies, either systems or python libraries,
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
conversations are lost as soon as you restart Rasa Core.
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
to add this store to Rasa Core:

  - extending the Rasa Core image
  - mounting it as volume

Then add the required configuration to your endpoint configuration
``config/endpoints.yml`` as it is described in :ref:`tracker_store`.
If you want the tracker store component (e.g. a certain database) to be part
of your docker compose file, add a corresponding service and configuration
there.

.. include:: feedback.inc
