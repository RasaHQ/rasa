:desc: Run and ship your Rasa assistant with Docker containers on any
       Docker-compatible machine or cluster.

.. _running-rasa-with-docker:

Running Rasa with Docker
========================

This is a guide on how to build a Rasa assistant with Docker.
If you haven't used Rasa before, we'd recommend that you start with the :ref:`rasa-tutorial`.

.. contents::
   :local:

Installing Docker
-----------------

If you're not sure if you have Docker installed, you can check by running:

  .. code-block:: bash

    docker -v && docker-compose -v
    # Docker version 18.09.2, build 6247962
    # docker-compose version 1.23.2, build 1110ad01

If Docker is installed on your machine, the output should show you your installed
versions of Docker and Docker Compose. If the command doesn't work, you'll have to
install Docker.
See `Docker Installation <https://docs.docker.com/install/>`_ for details.

Building an Assistant with Rasa and Docker
------------------------------------------

This section will cover the following:

    - Setting up your Rasa project and training an initial model
    - Talking to your AI assistant via Docker
    - Choosing a Docker image tag
    - Training your Rasa models using Docker
    - Talking to your assistant using Docker
    - Running a Rasa server with Docker


Setup
~~~~~

Just like in the :ref:`tutorial <rasa-tutorial>`, you'll use the ``rasa init`` command to create a project.
The only difference is that you'll be running Rasa inside a Docker container, using
the image ``rasa/rasa``. To initialize your project, run:

.. code-block:: bash

   docker run -v $(pwd):/app rasa/rasa init --no-prompt

What does this command mean?

- ``-v $(pwd):/app`` mounts your current working directory to the working directory
  in the Docker container. This means that files you create on your computer will be
  visible inside the container, and files created in the container will
  get synced back to your computer.
- ``rasa/rasa`` is the name of the docker image to run.
- the Docker image has the ``rasa`` command as its entrypoint, which means you don't
  have to type ``rasa init``, just ``init`` is enough.

Running this command will produce a lot of output. What happens is:

- a Rasa project is created
- an initial model is trained using the project's training data.

To check that the command completed correctly, look at the contents of your working directory:

.. code-block:: bash

   ls -1

The initial project files should all be there, as well as a ``models`` directory that contains your trained model.

Talking to Your Assistant
~~~~~~~~~~~~~~~~~~~~~~~~~

To talk to your newly-trained assistant, run this command:


.. code-block:: bash

   docker run -it -v $(pwd):/app rasa/rasa shell

This will start a shell where you can chat to your assistant.
Note that this command includes the flags ``-it``, which means that you are running
Docker interactively, and you are able to give input via the command line.
For commands which require interactive input, like ``rasa shell`` and ``rasa interactive``,
you need to pass the ``-it`` flags.


Customizing your Model
----------------------

Choosing a Tag
~~~~~~~~~~~~~~

To keep images as small as possible, we publish different tags of the ``rasa/rasa`` image
with different dependencies installed. See :ref:`choosing-a-pipeline` for more information
about depedencies.

All tags start with a version -- the ``latest`` tag corresponds to the current master build.
The tags are:

- ``{version}``
- ``{version}-spacy-en``
- ``{version}-spacy-de``
- ``{version}-mitie-en``
- ``{version}-full``

The plain ``{version}`` tag includes all the dependencies you need to run the ``supervised_embeddings`` pipeline.
If you are using components with pre-trained word vectors, you need to choose the corresponding tag.
Alternatively, you can use the ``-full`` tag, which includes all pipeline dependencies.

.. note::

   You can see a list of all the versions and tags of the Rasa Docker image
   `here <https://hub.docker.com/r/rasa/rasa/>`_.


.. _model_training_docker:

Training a Custom Rasa Model with Docker
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Edit the ``config.yml`` file to use the pipeline you want, and place
your NLU and Core data into the ``data/`` directory.
Now you can train your Rasa model by running:

.. code-block:: bash

  docker run \
    -v $(pwd):/app \
    rasa/rasa:latest-full \
    train \
      --domain domain.yml \
      --data data \
      --out models

Here's what's happening in that command:

  - ``-v $(pwd):/app``: Mounts your project directory into the Docker
    container so that Rasa can train a model on your training data
  - ``rasa/rasa:latest-full``: Use the Rasa image with the tag ``latest-full``
  - ``train``: Execute the ``rasa train`` command within the container. For more
    information see :ref:`command-line-interface`.

In this case, we've also passed values for the location of the domain file, training
data, and the models output directory to show how these can be customized.
You can also leave these out since we are passing the default values.

.. note::

    If you are using a custom NLU component or policy, you have to add the module file to your
    Docker container. You can do this by either mounting the file or by including it in your
    own custom image (e.g. if the custom component or policy has extra dependencies). Make sure
    that your module is in the Python module search path by setting the
    environment variable ``PYTHONPATH=$PYTHONPATH:<directory of your module>``.


Running the Rasa Server
-----------------------

To run your AI assistant in production, configure your required
:ref:`messaging-and-voice-channels` in ``credentials.yml``. If this file does not
exist, create it using:

.. code-block:: bash

  touch credentials.yml

Then edit it according to your connected channels.
After, run the trained model with:

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
    :ref:`command-line-interface`.


Using Docker Compose to Run Multiple Services
---------------------------------------------

To run Rasa together with other services, such as a server for custom actions, it is
recommend to use `Docker Compose <https://docs.docker.com/compose/>`_.
Docker Compose provides an easy way to run multiple containers together without
having to run multiple commands.

Start by creating a file called ``docker-compose.yml``:

.. code-block:: bash

  touch docker-compose.yml

Add the following content to the file:

.. code-block:: yaml

  version: '3.0'
  services:
    rasa:
      image: rasa/rasa:latest-full
      ports:
        - 5005:5005
      volumes:
        - ./:/app
      command:
        - run


The file starts with the version of the Docker Compose specification that you
want to use.
Each container is declared as a ``service`` within the docker-compose file.
The first service is the ``rasa`` service.

The command is similar to the ``docker run`` command.
The ``ports`` part defines a port mapping between the container and your host
system. In this case it makes ``5005`` of the ``rasa`` service available on
port ``5005`` of your host.
This is the port of the :ref:`REST Channel <rest_channels>` interface of Rasa.

.. note::

    Since Docker Compose starts a set of Docker containers, it is no longer
    possible to connect to the command line of a single container after executing the
    ``run`` command.

To run the services configured in your ``docker-compose.yml`` execute:

.. code-block:: bash

    docker-compose up


Adding Custom Actions
---------------------

To create more sophisticated assistants, you will want to use :ref:`custom-actions`.
Continuing the example from above, you might want to add an action which tells
the user a joke to cheer them up.

Creating a Custom Action
~~~~~~~~~~~~~~~~~~~~~~~~

Start by creating the custom actions in a directory ``actions``:

.. code-block:: bash

  mkdir actions
  # Rasa SDK expects a python module.
  # Therefore, make sure that you have this file in the directory.
  touch actions/__init__.py
  touch actions/actions.py

Then build a custom action using the Rasa SDK, e.g.:

.. code-block:: python

  import requests
  import json
  from rasa_sdk import Action


  class ActionJoke(Action):
    def name(self):
      return "action_joke"

    def run(self, dispatcher, tracker, domain):
      request = requests.get('http://api.icndb.com/jokes/random').json()  # make an api call
      joke = request['value']['joke']  # extract a joke from returned json response
      dispatcher.utter_message(joke)  # send the message back to the user
      return []

Next, add the custom action in your stories and your domain file.
Continuing with the example bot from ``rasa init``, replace ``utter_cheer_up`` in
``data/stories.md`` with the custom action ``action_joke``, and add
``action_joke`` to the actions in the domain file.

Adding the Action Server
~~~~~~~~~~~~~~~~~~~~~~~~

The custom actions are run by the action server.
To spin it up together with the Rasa instance, add a service
``action_server`` to the ``docker-compose.yml``:

.. code-block:: yaml
   :emphasize-lines: 11-14

   version: '3.0'
   services:
     rasa:
       image: rasa/rasa:latest-full
       ports:
         - 5005:5005
       volumes:
         - ./:/app
       command:
         - run
     action_server:
       image: rasa/rasa_sdk:latest
       volumes:
         - ./actions:/app/actions

This pulls the image for the Rasa SDK which includes the action server,
mounts your custom actions into it, and starts the server.

To instruct Rasa to use the action server you have to tell Rasa its location.
Add this to your ``endpoints.yml`` (if it does not exist, create it):

.. code-block:: yaml

  action_endpoint:
    url: http://action_server:5055/webhook

Run ``docker-compose up`` to start the action server together
with Rasa.

Adding Custom Dependencies
~~~~~~~~~~~~~~~~~~~~~~~~~~

If your custom action has additional dependencies of systems or Python libraries,
you can add these by extending the official image.

To do so, create a file named ``Dockerfile`` in which you extend the official
image and add your custom dependencies. For example:

.. code-block:: docker

    # Extend the official Rasa SDK image
    FROM rasa/rasa_sdk:latest

    # Add a custom system library (e.g. git)
    RUN apt-get update && \
        apt-get install -y git

    # Add a custom python library (e.g. jupyter)
    RUN pip install --no-cache-dir jupyter

You can then build the image via the following command, and use it in your
``docker-compose.yml`` instead of the ``rasa/rasa_sdk`` image.

.. code-block:: bash

  docker build . -t <name of your custom image>:<tag of your custom image>

Adding a Custom Tracker Store
-----------------------------

By default, all conversations are saved in memory. This means that all
conversations are lost as soon as you restart the Rasa server.
If you want to persist your conversations, you can use a different
:ref:`Tracker Store <tracker-stores>`.

Using PostgreSQL as Tracker Store
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Start by adding PostgreSQL to your docker-compose file:

.. code-block:: yaml

  postgres:
    image: postgres:latest

Then add PostgreSQL to the ``tracker_store`` section of your endpoint
configuration ``config/endpoints.yml``:

.. code-block:: yaml

  tracker_store:
    type: sql
    dialect: "postgresql"
    url: postgres
    db: rasa

Using MongoDB as Tracker Store
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Start by adding MongoDB to your docker-compose file. The following example
adds the MongoDB as well as a UI (you can skip this), which will be available
at ``localhost:8081``. Username and password for the MongoDB instance are
specified as ``rasa`` and ``example``.

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
configuration ``endpoints.yml``:

.. code-block:: yaml

  tracker_store:
    type: mongod
    url: mongodb://mongo:27017
    username: rasa
    password: example

Then start all components with ``docker-compose up``.

Using Redis as Tracker Store
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Start by adding Redis to your docker-compose file:

.. code-block:: yaml

  redis:
    image: redis:latest

Then add Redis to the ``tracker_store`` section of your endpoint
configuration ``endpoints.yml``:

.. code-block:: yaml

  tracker_store:
    type: redis
    url: redis

Using a Custom Tracker Store Implementation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you have a custom implementation of a tracker store you have two options
to add this store to Rasa:

  - extending the Rasa image
  - mounting it as volume

Then add the required configuration to your endpoint configuration
``endpoints.yml`` as it is described in :ref:`tracker-stores`.
If you want the tracker store component (e.g. a certain database) to be part
of your Docker Compose file, add a corresponding service and configuration
there.
