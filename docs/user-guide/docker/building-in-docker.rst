.. _building-in-docker:

Building an Assistant with Rasa and Docker
==========================================

If you don't have a Rasa project yet, you can build one in Docker without having to install Rasa Open Source
on your local machine. If you already have a model you're satisfied with, skip ahead to :ref:`running-the-rasa-server`
to deploy your model.

.. contents::
   :local:

Installing Docker
*****************

If you're not sure if you have Docker installed, you can check by running:

  .. code-block:: bash

    docker -v && docker-compose -v
    # Docker version 18.09.2, build 6247962
    # docker-compose version 1.23.2, build 1110ad01

If Docker is installed on your machine, the output should show you your installed
versions of Docker and Docker Compose. If the command doesn't work, you'll have to
install Docker.
See `Docker Installation <https://docs.docker.com/install/>`_ for details.

Setting up your Rasa Project
****************************

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


.. note::

   By default Docker runs containers as user ``1001``. Hence, all files created by
   these containers will be owned by this user. See the `documentation of docker
   <https://docs.docker.com/v17.12/edge/engine/reference/commandline/run/>`_
   and `docker-compose <https://docs.docker.com/compose/compose-file/>`_ if you want to
   run the containers as a different user.

Talking to Your Assistant
*************************

To talk to your newly-trained assistant, run this command:


.. code-block:: bash

   docker run -it -v $(pwd):/app rasa/rasa shell

This will start a shell where you can chat to your assistant.
Note that this command includes the flags ``-it``, which means that you are running
Docker interactively, and you are able to give input via the command line.
For commands which require interactive input, like ``rasa shell`` and ``rasa interactive``,
you need to pass the ``-it`` flags.


Customizing your Model
**********************

Choosing a Tag
##############

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
########################################

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

Adding Custom Actions
*********************

To create more sophisticated assistants, you will want to use :ref:`custom-actions`.
Continuing the example from above, you might want to add an action which tells
the user a joke to cheer them up.

Running a Custom Action in Docker-Compose
#########################################

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
      dispatcher.utter_message(text=joke)  # send the message back to the user
      return []

Next, add the custom action in your stories and your domain file.
Continuing with the example bot from ``rasa init``, replace ``utter_cheer_up`` in
``data/stories.md`` with the custom action ``action_joke``, and add
``action_joke`` to the actions in the domain file.

To instruct Rasa to use the action server you have to tell Rasa its location.
Add this to your ``endpoints.yml`` (if it does not exist, create it):

.. code-block:: yaml

  action_endpoint:
    url: http://app:5055/webhook

To spin up the action server together with the Rasa instance, add a service
``app`` to the ``docker-compose.yml``:

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
     app:
       image: rasa/rasa-sdk:latest
       volumes:
         - ./actions:/app/actions

This pulls the image for the Rasa SDK which includes the action server,
mounts your custom actions into it, and starts the server.

Run ``docker-compose up`` to start the action server together with Rasa.

.. warning::

   If you create a more complicated
   action that has extra library dependencies, you will need to
   :ref:`build an action server image<building-an-action-server-image>` to run your code.

Deploying your Model
####################

Once you're happy with your model, you can