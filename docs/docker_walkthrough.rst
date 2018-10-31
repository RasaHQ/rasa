.. _docker_walkthrough:

Docker Walkthrough
==================

This walkthrough provides a tutorial on how to set up Rasa Core, Rasa NLU,
an Action Server, and Duckling with Docker containers. The tutorial starts from
scratch. Hence, all you need is a bit of familiarity with your command line.

Let's get started!

.. contents::

1. Setup
--------

In order to get going we need only two things installed.

    - A text editor of your choice
    - Docker

If you do not have installed `Docker <https://www.docker.com/>`_ so far, you
have to install it, too. If you are not sure whether it is installed execute the
following command:

  .. code-block:: bash

    docker -v && docker-compose -v
    # Docker version 18.06.1-ce, build e68fc7a
    # docker-compose version 1.22.0, build f46880f

If your output is not similar to the one above, please install Docker.
See `this instruction page <https://docs.docker.com/install/>`_
of the Docker documentation which describes how to install Docker on a variety
of operating systems.

Congratulation! 🚀 Now we have got everything set up to get going!

2. Create a simple chatbot using Rasa Core
------------------------------------------

In this part we will

    - create some story lines for our chatbot
    - train our chatbot
    - run our chatbot within a Docker container

2.1 Create Stories
~~~~~~~~~~~~~~~~~~~~~~~~

Rasa Core works by learning from example conversations. Therefore, we now need
to write a few examples to get started.
Stories are written in `Markdown <https://en.wikipedia.org/wiki/Markdown>`_
and follow a simple guideline.

.. code-block:: md

  ## Story 1        <-- the name of your story
  * greet           <-- the intent of the person talking to the chatbot
    - utter_greet   <-- a template of your bot's response

Before we start writing stories, we have to create a few directories.
First, a directory ``project-dir`` which will contain all the files you are
creating in this tutorial. Within the ``project-dir`` we further create a
directory ``data`` which will contain our training data.
Then we can finally create a file `stories.md` in the ``data`` directory in
which we write our stories.

Using the command line:

.. code-block:: bash

  # create a project-dir called `docker-tutorial` and the `data` directory
  mkdir -p docker-tutorial/data

  # change to your project directory
  cd docker-tutorial

  # create a file which will contain your stories
  touch data/stories.md

After these steps, your directory structure should roughly look like this:

.. code-block:: bash

  docker-tutorial       <-- the `project-dir`
  └── data              <-- the `data` directory which will contain your training data
      └── stories.md    <-- the file which contains your stories

For a start let's build a bot which can say `Hi`, asks for your mood and then
gives a different answer depending on your input. Put these basic stories
in the file ``data/stories.md``:

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

2.2 Create the Bot's Universe
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

After defining some training data for our bot, we have to define its domain.
You can think about the domain as the world your bots live in and contains all
capabilities your bots has.

To handle our examples stories, our bot has to have the following capabilites:

  - recognize user intents

    - saying hello to the bot (`greet`)
    - saying that they are happy (`mood_happy`)
    - saying that they are sad (`mood_unhappy`)
    - saying goodbye to the bot (`good_bye`)

  - respond to the user input

    - response for happy users (`utter_happy`)
    - cheer up a sad user (`utter_cheer_up`)
    - say goodbye (`utter_goodbye`)

.. note::

  By convention the templates for bot responses have the prefix ``utter_``.
  You can follow this convention but don't have to.

The domain file is used to stores all these capabilities.
Further, we put templates for our bot's responses in it.

Let's start by creating the domain file:

.. code-block:: bash

  touch domain.yml

Then we put our required capabilities in the created ``domain.yml``:

.. code-block:: yaml

    intents:            # <-- intents of the user speaking to your bot
      - greet
      - mood_happy
      - mood_unhappy
      - goodbye

    actions:            # <-- actions your bot can execute in response to user input
      - utter_greet
      - utter_happy
      - utter_cheer_up
      - utter_goodbye

    templates:          # <-- templates for your bot's answers
      utter_greet:
        - text: "Hi, how is it going?"
      utter_happy:
        - text: "Great, carry on!"
      utter_cheer_up:
        - text: "Here is something to cheer you up:"
          image: "https://i.imgur.com/nGF1K8f.jpg"
      utter_goodbye:
        - text: "Goodbye!"

2.3 Training our Bot
~~~~~~~~~~~~~~~~~~~~

We now have everything set up to train our bot! Hooray!
To so, we have to mount the training data and the domain file in the Rasa Core
container.

.. code-block:: bash

  docker run \
    -v $(pwd):/app/project \
    -v $(pwd)/models/rasa_core:/app/models \
    rasa/rasa_core:latest \
    train \
      --domain project/domain.yml \
      --stories project/data/stories.md \
      --out models

Yey! You just trained your chatbot for the first time!
In case you are wondering what is going on the Docker command above, here is
some explanation:

  - ``-v $(pwd):/app/project``: Mounts your `project-dir` into the Docker
    container so that the bot can be trained on your story data and the domain
    file
  - ``-v $(pwd)/models/rasa_core:/app/models``: Mounts a directory
    `project-dir/models/rasa_core` in the container which is used to store the
    trained Rasa Core model. You should see this directory on your host after
    the training!
  - ``rasa/rasa_core:latest``: Use the latest Rasa Core Docker image
  - ``train``: Execute the ``train`` command within the container with

    - ``--domain project/domain.yml``: Path to your domain file from within the
      container
    - ``--stories project/data/stories.md``: Path to your training stories from
      within the container
    - ``--out models``: Instructs Rasa Core to store the trained model in the
      directory ``models`` which corresponds to your host directory
      ``models/rasa_core``

2.4 Testing our Bot
~~~~~~~~~~~~~~~~~~~

Now that we have trained our bot, we also want to test it out. Note, that we
have not yet connected Rasa NLU to our chatbot so it will not yet understand
human language and therefore not be able to understand the user's intent.
Therefore, we have to specify the user's intent directly by input the name
of the intent preceded by a ``/``. Confused? Let me show you how to do that👇

.. code-block:: bash

  docker run \
    -it \
    -v $(pwd)/models/rasa_core:/app/models \
    rasa/rasa_core:latest \
    start \
      --core models

A bit of explanation on the single parts of the commands:

  - ``-it``: Runs the Docker container in interactive mode so that you can
    interact with the console of the container
  - ``-v $(pwd)/models/rasa_core:/app/models``: Mounts the trained Rasa Core
    model in the container
  - ``rasa/rasa_core:latest``: We are using the same image as before
  - ``start``: Executes the start command which connects to the bot on the
    command line with

    - ``--core models``: Defines the location of the trained model which is
      used for the conversation.

After executing the command above you should see a line saying
`Bot loaded. Type a message and press enter (use '/stop' to exit):`.
We now start by greeting the bot. Type `/greet` into your commandline.
Your bot should now utter `Hi, how is it going?`.
From there you can go on, e.g. by inputting `mood_happy`.

.. code-block:: bash
  :emphasize-lines: 2,4,6

  /greet
  Hi, how is it going?
  /mood_happy
  Great, carry on!
  /goodbye
  Bye

Congrats! You just had your first conversations with your bot!

3. Adding Natural Language Understanding (Rasa NLU)
---------------------------------------------------

3.1 Add NLU Training Data
~~~~~~~~~~~~~~~~~~~~~~~~~

After your first conversations with the bot you might be thinking
`But I want to have REAL conversations with my bot`.
Don't worry, we will now teach our bot how to understand human language!

Similar to Rasa Core, we have to teach our bot by example.
Therefore, we create a file ``data/nlu.md`` which contains examples for each
user intent.

.. code-block:: bash

  touch data/nlu.md

Open the created file and add the following content:

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
  - extremly sad
  - so sad

  ## intent:goodbye
  - bye
  - goodbye
  - see you around
  - see you later

3.1 Train the NLU Model
~~~~~~~~~~~~~~~~~~~~~~~

We are ready to train the Rasa NLU model!
Copy the command below and execute it on your commandline:

.. code-block:: bash

  docker run \
    -v $(pwd):/app/project \
    -v $(pwd)/models/rasa_nlu:/app/models \
    rasa/rasa_nlu:latest-spacy \
    run \
      python -m rasa_nlu.train \
      -c config.yml \
      -d project/data/nlu.md \
      -o models \
      --project current

A quick explanation of the used command:

  - ``-v $(pwd):/app/project``: Mounts your project-dir into the Docker
    container so that the bot can be trained on your NLU data.
  - ``-v $(pwd)/models/rasa_nlu:/app/models``: Mounts the directory
    ``project-dir/models/rasa_nlu`` in the container which is used to store the
    trained Rasa NLU model. You should see this directory on your host after
    the training!
  - ``rasa/rasa_nlu:latest-spacy``: We are using the latest Rasa NLU which uses
    the `spaCy` `pipeline <https://rasa.com/docs/nlu/choosing_pipeline/>`_ .
    This pipeline is good if you have a bot in english and not much training
    data.
  - ``run``: Entrypoint parameter to run any command within the NLU container
  - ``python -m rasa_nlu.train``: Starts the NLU training with

    - ``-c config.yml``: Uses the default NLU pipeline configuration which is
      provided by the Docker image
    - ``-d project/data/nlu.md``: Path to our NLU trainings data
    - ``-o models``: The directory which is used to store the nlu models
    - ``--project current``: The project name to use
      ``models/rasa_nlu/nlu``

3.2 Connect Rasa Core and Rasa NLU
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Now let's connect Rasa Core and Rasa NLU. One option would be to start
each container individually. However, this setup can get quite complicated
as soon as we add more components. We suggest using
`docker compose <https://docs.docker.com/compose/>`_.
Docker Compose uses a description in a
`yaml <https://en.wikipedia.org/wiki/YAML>`_ file which let's us specify
all components and their configuration. Using this we can start everything
using a single command!

Let's create the compose file:

.. code-block:: bash

  touch docker-compose.yml

At the top of the file we put the version of the Docker Compose specification
we want to use. In our case:

.. code-block:: yaml

  version: '3.0'

Then we define the so called ``services`` which means the configuration of the
containers we want to start. Let's start with the Rasa Core service:

.. code-block:: yaml

  services:
    rasa_core:
      image: rasa/rasa_core:latest
      ports:
        - 5005:5005
      volumes:
        - ./models/rasa_core:/app/models
      command:
        - start
        - --core
        - models
        - -c
        - rest

The configuration will look pretty familiar to you apart from ``ports`` and
the start up parameters ``-c rest``. The ``ports`` part defines port mapping
between the container and your host system. In this case we make port ``5005`
of the service (aka container) ``rasa_core`` available on port ``5005` of our
host. This is the port of the :ref:`rest_channels` interface of Rasa Core.
We instruct Rasa Core to use REST as input / output channel by the run command
``-c rest``. Why are we not longer using the command line interface?
Since Docker Compose starts a whole bunch of Docker containers we cannot longer
directly connect to one single container after executing the ``run`` command.

The next service we are adding is Rasa NLU:

.. code-block:: yaml

  rasa_nlu:
      image: rasa/rasa_nlu:latest-spacy
      volumes:
        - ./models/rasa_nlu:/app/models
      command:
        - start
        - --path
        - models

Cool, so we translated our ``docker run`` commands according to the
docker-compose specification. While we it would be already to possible to
run the containers in this configuration, one part is still missing: we have
to instruct Rasa Core to connect to Rasa NLU to parse the user messages.
To do so, we create a file called ``config/endpoints.yml``:

.. code-block:: bash

  mkdir config
  touch config/endpoints.yml

Docker containers which are started using Docker Compose are using the same
network. Hence, each service can access other services by their service name.
E.g., ``rasa_core`` can access ``rasa_nlu`` using ``rasa_nlu`` as domain name.
Therefore, we put the following content in ``config/endpoints.yml``.

.. code-block:: yaml

  nlu:
    url: http://rasa_nlu:5000

Furthermore, we have to mount the ``config`` directory into the Rasa Core
container. Also we need to instruct Rasa Core to use this configuration as
well as specify the NLU model to use. These is done with the
``--endpoints <path to endpoints.yml>`` parameter and
``-u <nlu model to use>``. By adding these to our ``docker-compose.yml`` it
should have the following content:

.. code-block:: yaml

  version: '3.0'

  services:
    rasa_core:
      image: rasa/rasa_core:latest
      ports:
        - 5005:5005
      volumes:
        - ./models/rasa_core:/app/models
        - ./config/:/app/config
      command:
        - start
        - --core
        - models
        - -c
        - rest
        - --endpoints
        - config/endpoints.yml
        - -u
        - current/
    rasa_nlu:
      image: rasa/rasa_nlu:latest-spacy
      volumes:
        - ./models/rasa_nlu:/app/models
      command:
        - start
        - --path
        - models

Well done! We have got everything hooked up now and can finally talk to our
bot!

3.3 Speak to the Bot
--------------------

The REST API of Rasa Core is now available on ``http://localhost:5005``.
To send messages to your bot, try:

.. code-block:: bash

  curl --request POST \
    --url http://localhost:5005/webhooks/rest/webhook \
    --header 'content-type: application/json' \
    --data '{
      "message": "hello"
    }'

Your bot should then answer something like:

.. code-bock:: bash

  [
    {
      "recipient_id": "default",
      "text": "Hi, how is it going?"
    }
  ]

If the bot cannot understand you (which is quite like since we provided only
very little NLU training data), the answer is something along the lines:

.. code-block:: bash

  []

Hooray! 🎉 🎊 We can now train Rasa Core, Rasa NLU and speak to our bot in
human language!

4. Adding Custom Actions
========================

5. Adding Duckling
==================

6. Adding a Custom Tracker Store
================================




