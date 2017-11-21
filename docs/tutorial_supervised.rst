.. _tutorial_supervised:

Supervised Learning Tutorial
============================

.. testsetup::

   import os, sys
   p = os.path.abspath(os.path.join('examples', 'restaurantbot'))
   os.chdir(p)
   sys.path.append(p)

.. note::

   This tutorial will cover how to use Rasa Core directly from python. We will
   dive a bit deeper into the different concepts and overall structure of the
   library. You should already be familiar with the terms domain, stories, and
   have some knowledge of NLU.

   `Example Code on GitHub <https://github.com/RasaHQ/rasa_core/tree/master/examples/restaurantbot>`_

Goal
^^^^

In this example we will create a restaurant search bot, by training 
a neural net on example conversations. A user can contact the bot with something
close to ``"I want a mexican restaurant!"`` and the bot will ask more details
until it is ready to suggest a restaurant.

Let's start by creating a new project directory:

.. code-block:: bash

   mkdir restaurantbot && cd restaurantbot

After we are done creating the different files, the final structure will look
like this:

.. code-block:: text

   restaurantbot/
   ├── data/
   │   ├── babi_stories.md       # dialogue training data
   │   └── franken_data.json     # nlu training data
   ├── restaurant_domain.yml     # dialogue configuration
   └── nlu_model_config.json     # nlu configuration


All example code snippets assume you are running the code from within that
project directory.

1. Creating the Domain
^^^^^^^^^^^^^^^^^^^^^^

Our restaurant domain contains a couple of slots as well as a number of
intents, entities, utterance templates and actions. Let's save the following
domain definition in ``restaurant_domain.yml``:

.. literalinclude:: ../examples/restaurantbot/restaurant_domain.yml
    :linenos:

You can instantiate that ``Domain`` like this:

.. testcode::

    from rasa_core.domain import TemplateDomain
    domain = TemplateDomain.load("restaurant_domain.yml")

Our ``Domain`` has clearly defined ``slots`` (in our case criterion for target restaurant) and ``intents``
(what the user can send). It also requires ``templates`` to have text to use to respond given a certain ``action``.

Each of these ``actions`` must either be named after an utterance (dropping the ``utter_`` prefix)
or must be a module path to an action. Here is the code for one the two custom actions:

.. testcode::

    from rasa_core.actions import Action

    class ActionSearchRestaurants(Action):
        def name(self):
            return 'search_restaurants'

        def run(self, dispatcher, tracker, domain):
            dispatcher.utter_message("here's what I found")
            return []


The ``name`` method is to match up actions to utterances, and the ``run`` command is run whenever the action is called. This
may involve api calls or internal bot dynamics.

But a domain alone doesn't make a bot, we need some training data to tell the
bot which actions it should execute at what point in time. So lets create some
stories!

2. Creating Training Data
^^^^^^^^^^^^^^^^^^^^^^^^^

The training conversations come from the `bAbI dialog task <https://research.fb.com/downloads/babi/>`_ . 
However, the messages in these dialogues are machine generated, so we will augment 
this dataset with real user messages from the `DSTC dataset <http://camdial.org/~mh521/dstc/>`_.
Lucky for us, this dataset is also in the restaurant domain. 

.. note:: 
   the babi dataset is machine-generated, and there are a LOT of dialogues in there. 
   There are 1000 stories in the training set, but you don't need that many to build
   a useful bot. How much data you need depends on the number of actions you define, and the 
   number of edge cases you want to support. But a few dozen stories is a good place to start.

We have converted the bAbI dialogue training set into the Rasa stories format, you
can download the stories training data from `GitHub <https://raw.githubusercontent.com/RasaHQ/rasa_core/master/examples/restaurantbot/data/babi_stories.md>`_.
Download that file, and store it in ``babi_stories.yml``. Here's an example
conversation snippet:

.. code-block:: md
    :linenos:

    ## story_07715946
    * _greet[]
     - action_ask_howcanhelp
    * _inform[location=rome,price=cheap]
     - action_on_it
     - action_ask_cuisine
    * _inform[cuisine=spanish]
     - action_ask_numpeople
    * _inform[people=six]
     - action_ack_dosearch
     ...

See :ref:`stories` to get more information about the Rasa Core data format.
We can also visualize that training data to generate a graph which is similar to a flow chart:

.. image:: _static/images/babi_flow.png

The graph shows all of the actions executed in the training data, and the user messages (if any) 
that occurred between them. As you can see, flow charts get complicated quite quickly. Nevertheless, they
can be a helpful tool in debugging a bot. More information can be found in :ref:`story-visualization`.

3. Training your bot
^^^^^^^^^^^^^^^^^^^^

We can go directly from data to bot with only a few steps:

1. train a Rasa NLU model to extract intents and entities. Read more about that in the `NLU docs <http://rasa-nlu.readthedocs.io/>`_.
2. train a dialogue policy which will learn to choose the correct actions
3. set up an agent which has both model 1 and model 2 working together to go directly from **user input** to **action**

We will go through these steps one by one.

NLU model
---------

To train our Rasa NLU model, we need to create a configuration first in ``config_nlu.json``:

.. literalinclude:: ../examples/restaurantbot/nlu_model_config.json
   :linenos:

We can train the NLU model using

.. code-block:: bash

   python -m rasa_nlu.train -c nlu_model_config.json --fixed_model_name current

or using python code

.. literalinclude:: ../examples/restaurantbot/bot.py
   :linenos:
   :pyobject: train_nlu


You can learn all about Rasa NLU starting from the
`github repository <https://github.com/RasaHQ/rasa_nlu>`_.
What you need to know though is that ``interpreter.parse(user_message)`` returns
a dictionary with the intent and entities from a user message.


*Training NLU takes approximately 18 seconds on a 2014 MacBook Pro.*

Dialogue Policy
---------------

Now our bot needs to learn what to do in response to these messages. 
We do this by training one or multiple Rasa Core policies.

For this bot, we came up with our own policy. Here are the gory details:

.. literalinclude:: ../examples/restaurantbot/bot.py
   :linenos:
   :pyobject: RestaurantPolicy

.. note::
   Remember, you do not need to create your own policy. The default policy setup
   using a memoization policy and a Keras policy works quite well. Nevertheless,
   you can always fine tune them for your use case. Read :ref:`plumbing` for more info.

This policy extends the Keras Policy modifying the ML architecture of the
network. The parameters ``max_history_len`` and ``n_hidden`` may be altered
dependent on the task complexity and the amount of data one has.
``max_history_len`` is important as it is the amount of story steps the
network has access to to make a classification.

Now let's train it:

.. literalinclude:: ../examples/restaurantbot/bot.py
   :linenos:
   :pyobject: train_dialogue

This code creates the policies to be trained and uses the story training data
to train and persist a model. The goal of the trained policy is to predict
the next action, given the current state of the bot.

To train it from the cmd, run

.. code-block:: bash

   python bot.py train-dialogue

to get our trained policy.

.. testsetup::

   from bot import train_dialogue
   train_dialogue()


*Training the dialogue model takes roughly 12 minutes on a 2014 MacBook Pro*

4. Using the bot
^^^^^^^^^^^^^^^^

Now we're going to glue some pieces together to create an actual bot.
We instantiate the policy, and an ``Agent`` instance,
which owns an ``Interpreter``, a ``Policy``, and a ``Domain``.

We put the NLU model into an ``Interpreter`` and then put that into an ``Agent``.

We will pass messages directly to the bot, but this is just for
this is just for demonstration purposes. You can look at how to
build a command line bot and a facebook bot by checking out the :ref:`connectors`.

.. doctest::

   from rasa_core.interpreter import RasaNLUInterpreter
   from rasa_core.agent import Agent

   agent = Agent.load("models/dialogue", interpreter=RasaNLUInterpreter("models/nlu/current"))

We can then try sending it a message:

.. doctest::

   >>> agent.handle_message("_greet")
   [u'hey there!']

And there we have it! A minimal bot containing all the important pieces of Rasa Core.

If you want to handle input from the command line (or a different input channel) you need handle
that channel instead of handling messages directly, e.g.:

.. code-block:: python

   from rasa_core.channels.console import ConsoleInputChannel
   agent.handle_channel(ConsoleInputChannel())

In this case messages will be retrieved from the command line because we specified
the ``ConsoleInputChannel``. Responses are printed to the command line as well. You
can find a complete example of how to load an agent and chat with it on the
command line in ``examples/restaurantbot/run.py``.
