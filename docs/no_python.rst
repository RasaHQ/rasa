.. _no_python:

But I don't code in python!
===========================


While python is the *lingua franca* of machine learning, we're aware
that most chatbots are built in javascript, and that many enterprises are 
more comfortable building & shipping applications in java, C#, etc. 

We've made every effort to make sure that you can **still** use Rasa Core
even if you don't want to use python. However, do consider that Rasa Core
is a *framework*, and doesn't fit into a REST API as easily as Rasa NLU does. 



Rasa Core with minimal Python
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

You can build a chatbot with Rasa Core by:

* defining a domain (:ref:`a yaml file <domain>`)
* writing / collecting stories (:ref:`markdown format <stories>`)
* running python scripts to train and run your bot

The only part where you need to write python is when you want to define custom actions. 
There's an excellent python library called `requests <http://docs.python-requests.org/en/master/>`_, which makes HTTP programming painless.
If Rasa just needs to interact with your other services over HTTP, your actions will all look 
something like this:


.. doctest::

   from rasa_core.actions import Action
   import requests

   class ApiAction(Action):
       def name(self):
           return "my_api_action"

       def run(self, dispatcher, tracker, domain):
           data = requests.get(url).json
           return [SlotSet("api_result", data)]

Rasa Core with Docker
^^^^^^^^^^^^^^^^^^^^^

We provide a Dockerfile which allows you to build an image of Rasa Core
with a simple command: ``docker build -t rasa_core .``

The default command of the resulting container starts the Rasa Core server
with the ``--core`` and ``--nlu`` options. At this stage the container does not
yet contain any models, so you have to mount them from a local folder into
the container's ``/app/model/dialogue`` and ``app/model/nlu`` directories.
The full run command looks like this:

.. code-block:: bash

   docker run \
      --mount type=bind,source=<PATH_TO_DIALOGUE_MODEL_DIR>,target=/app/model/dialogue \
      --mount type=bind,source=<PATH_TO_NLU_MODEL_DIR>,target=/app/model/nlu \
      rasa_core

You also have the option to use the container to train a model with

.. code-block:: bash

   docker run \
      --mount type=bind,source=<PATH_TO_STORIES_FILE>/stories.md,target=/app/stories/stories.md \
      --mount type=bind,source=<PATH_TO_DOMAIN_FILE>/domain.yml,target=/app/domain/domain.yml \
      --mount type=bind,source=<OUT_PATH>,target=/app/out \
      rasa_core train

You may in addition run any Rasa Core command inside the container with
``docker run rasa_core run [COMMAND]``.

Rasa Core with ZERO Python
^^^^^^^^^^^^^^^^^^^^^^^^^^

If you are really constrained to not use any python, you can also use Rasa Core
through a :ref:`HTTP API <section_http>`.
