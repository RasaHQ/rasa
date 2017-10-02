.. _tutorial_babi:

Supervised Learning Tutorial
============================

Meta
^^^^

In this example we will create a restaurant search bot, by training 
a neural net on example conversations. 

A user can contact the bot with something close to "I want a mexican restaurant!"
and the bot will ask more details until it is ready to suggest a restaurant.

This assumes you already know what the ``Domain``, ``Policy``, and ``Action`` classes do. 
If you don't, it's a good idea to read the basic tutorial first. 

The Dataset
^^^^^^^^^^^

The training conversations come from the `bAbI dialog task <https://research.fb.com/downloads/babi/>`_ . 
However, the messages in these dialogues are machine generated, so we will augment 
this dataset with real user messages from the `DSTC dataset <http://camdial.org/~mh521/dstc/>`_.
Lucky for us, this dataset is also in the restaurant domain. 


.. note:: 
   the babi dataset is machine-generated, and there are a LOT of dialogues in there. 
   There are 1000 stories in the training set, but you don't need that many to build
   a useful bot. How much data you need depends on the number of actions you define, and the 
   number of edge cases you want to support. But a few dozen stories is a good place to start.


Here's an example conversation snippet: 

.. code-block:: md

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

You can read about the Rasa data format here : :ref:`stories`.
It may be worth browsing through ``data/babi_task5_trn_rasa_with_slots.md`` to get a sense of how these work.

We can also visualize that training data to generate a graph which is similar to a flow chart:

.. image:: _static/images/babi_flow.png

The chart shows the incoming user intents and entities and the action the bot is supposed to execute based on
the stories from the training data. As you can see, flow charts get complicated quite quickly. Nevertheless, they
can be a helpful tool in debugging a bot. More information can be found in :ref:`story-visualization`.

Training your bot
^^^^^^^^^^^^^^^^^

We can go directly from data to bot with only a few steps:

1. train a Rasa NLU model to extract intents and entities. Read more about that in the `NLU docs <http://rasa-nlu.readthedocs.io/>`_.
2. train a dialogue policy which will learn to choose the correct actions
3. set up an agent which has both model 1 and model 2 working together to go directly from **user input** to **action**

We will go through these steps one by one.

1. Train NLU model
------------------

Our ``train_nlu.py`` program looks like this:

.. literalinclude:: ../examples/babi/train_nlu.py
   :pyobject: train_babi_nlu

You can learn all about Rasa NLU starting from the
`github repository <https://github.com/RasaHQ/rasa_nlu>`_.
What you need to know though is that ``interpreter.parse(user_message)`` returns
a dictionary with the intent and entities from a user message.


*This step takes approximately 18 seconds on a 2014 MacBook Pro.*

2. Train Dialogue Policy
------------------------

Now our bot needs to learn what to do in response to these messages. 
We do this by training the Rasa Core model. From ``train_dm.py``:

.. literalinclude:: ../examples/babi/train_dm.py
   :pyobject: train_babi_dm


This creates a ``policy`` object. What you need to know is that ``policy.next_action`` chooses
which action the bot should take next.

Here we'll quickly explain the Domain and Policy objects, 
feel free to skip this if you don't care, or read :ref:`plumbing` for more info.

Domain
::::::

Let's start with ``Domain``. From ``restaurant_domain.yml``:

.. literalinclude:: ../examples/restaurant_domain.yml
   :language: yaml

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

Policy
::::::

From ``examples/restaurant_example.py`` again:

.. literalinclude:: ../examples/restaurant_example.py
   :pyobject: RestaurantPolicy


This policy builds an  LSTM in Keras  which will then be taken by the trainer and trained. The parameters ``max_history_len``
and ``n_hidden`` may be altered dependent on the task complexity and the amount of data one has. ``max_history_len`` is
important as it is the amount of story steps the network has access to to make a classification.

Training
::::::::

Now we can simply run ``python train_dm.py`` to get our trained policy.

*This step takes roughly 12 minutes on a 2014 MacBook Pro*


Using your bot
^^^^^^^^^^^^^^

Now we have a trained NLU and DM model which can be merged together to make a bot. This is done using an ``Agent``
object. From ``run.py``:

.. literalinclude:: ../examples/babi/run.py
   :language: python
   :pyobject: run_babi

We put the NLU model into an ``Interpreter`` and then put that into an ``Agent``.

You now have a working bot! It will recommend you the same place (papi's pizza place) no matter what preferences you
give, but at least its trying!