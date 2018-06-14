.. _tutorial_remote:

Rasa Core without Python
========================


.. note:: 
   In this tutorial we will build a demo app using Rasa Core as a HTTP server.
   Although, you do not need to write any python code - you still need to
   understand basic concepts like domains, stories, and actions.

   `Example Code on GitHub <https://github.com/RasaHQ/rasa_core/tree/master/examples/remotebot>`_

Goal
^^^^

We will create a simple bot that doesn't require you to write your custom action
code in python, but rather any other language. To enable that, the process is
a bit different for handling a message - as Rasa Core can not execute the
actions internally but needs to wait for you to execute them.

Let's start by creating a project folder:

.. code-block:: bash

   mkdir remotebot && cd remotebot

After this tutorial, the folder structure should look like this:

.. code-block:: text

   remotebot/
   ├── data/
   │   ├── stories.md              # dialogue training data
   │   └── concert_messages.md     # nlu training data
   ├── concert_domain_remote.yml   # dialogue configuration
   └── nlu_model_config.yml        # nlu configuration

The first steps of creating a bot are very similar to other Rasa Core bots.
But let's go through each of them anyway - we will get to the HTTP
interface at the end!

1. Define a Domain
^^^^^^^^^^^^^^^^^^

The domain is similar to other examples, but does not contain utter templates.
This is because you need to handle the connection to the input and output
yourself (e.g. sending and receiving messages from facebook).

Here is an example domain for our remotebot, ``concert_domain_remote.yml``:


.. literalinclude:: ../examples/remotebot/concert_domain_remote.yml
   :linenos:
   :language: yaml

One important difference is ``action_factory: remote``. This tells Rasa that
it is not supposed to run the actions on its own.

.. note::
   You can **choose whatever action names you like**. Rasa Core will not (and for
   that matter, can not) check their validity. Rasa will use these names to
   notify you about which action needs to be executed. So you need to be able
   to know what to do given an action name.

2. Define an interpreter
^^^^^^^^^^^^^^^^^^^^^^^^

We are going to use Rasa NLU as an interpreter, so let's create
some intent examples in ``data/concert_messages.md``:

.. literalinclude:: ../examples/remotebot/data/concert_messages.md
   :linenos:
   :language: md

Furthermore, we need a configuration file ``nlu_model_config.yml`` for the
NLU model:

.. literalinclude:: ../examples/remotebot/nlu_model_config.yml
   :language: yaml
   :linenos:

We can now train a NLU model using our examples (make sure to
`install Rasa NLU <http://nlu.rasa.com/installation.html#setting-up-rasa-nlu>`_
first as well as
`spaCy <http://nlu.rasa.com/installation.html#best-for-most-spacy-sklearn>`_).

Let's run

.. code-block:: bash

   python -m rasa_nlu.train -c nlu_model_config.yml --fixed_model_name current \
         --project nlu --path models --data data/concert_messages.md

to train our NLU model. A new directory ``models/nlu/current`` should have been
created containing the NLU model.

.. note::

   To gather more insights about the above configuration and Rasa NLU features
   head over to the `Rasa NLU documentation <https://nlu.rasa.com>`_.

3. Define stories
^^^^^^^^^^^^^^^^^

We need to add couple of stories as well to define the flow (so we can finally
get to the interesting part of running in remote mode). Let's put them into
``data/stories.md``:

.. literalinclude:: ../examples/remotebot/data/stories.md
   :linenos:
   :language: md

To train the dialogue model, run:

.. code-block:: bash

   python -m rasa_core.train -s data/stories.md -d concert_domain_remote.yml -o models/dialogue

This will train the model and store it into ``models/dialogue``.

4. Running the server
^^^^^^^^^^^^^^^^^^^^^

Now we can use our trained dialogue model and the previously created NLU model
to run our bot in server mode:

.. code-block:: bash

   python -m rasa_core.server -d models/dialogue -u models/nlu/current -o out.log

And there we have it! The bot is running and waiting for your HTTP requests.

5. Using the server
^^^^^^^^^^^^^^^^^^^

All communication is going to happen over a HTTP interface. You need to send
a request to that interface to start a conversation as well as for the actions
you have run on your end.

A detailed explanation including examples you can use with your running server
can be found in :ref:`http_start_conversation`.

.. raw:: html 
   :file: poll.html
