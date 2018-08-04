:desc: Interactive Learning with Rasa Core

.. _interactive_learning:

Interactive Learning
====================

.. note::

    We're using this `Example Code on GitHub <https://github.com/RasaHQ/rasa_core/tree/master/examples/concertbot>`_.


The Problem
^^^^^^^^^^^

Your bot usually has well-defined goals it should reach when talking to a
user. There are often numerous different ways the conversation could
develop before reaching this final stage. We'll teach you how to use
Rasa Core to bootstrap full-blown conversations from minimal to no training
data.


The Bot
^^^^^^^

Say you want to  build a bot that recommends concerts to go to. There is one
goal: you know that at the end of the conversation you want your bot to
make a recommendation. We'll show how to implement context-aware behaviour
without writing a flow chart. For example, if our user asks the question:
*which of those has better reviews?*, our bot should know whether they want
to compare *musicians* or *venues*.

Head over to ``examples/concertbot`` for this example. Let's go!

The Domain
^^^^^^^^^^

We will keep the concert domain simple, and won't add any slots just yet.
We'll also only support these intents:
``"greet", "thankyou", "goodbye", "search_concerts", "search_venues",
"compare_reviews"``. Here is the domain definition (``concert_domain.yml``):

.. literalinclude:: ../examples/concertbot/concert_domain.yml
    :linenos:

Stateless Stories
^^^^^^^^^^^^^^^^^

We start by training a stateless model on some simple dialogues in the
Rasa story format. This means we define conversations with
one user utterance and only a few (typically one) bot action in response. We
will use these stateless stories as a starting point for interactive learning.

In many cases, simple training 'conversations' are just a single turn and
response: "Hello" is always met with a greeting, "goodbye!" is always met
with a sign-off, and the correct response to "thank you" is pretty much
always "you're welcome".

Below is an excerpt of the stories.


.. note::

    Notice that below, we've defined two stories, showing that
    ``action_show_venue_reviews`` and ``action_show_concert_reviews``
    are both possible responses to the ``compare_reviews`` intent, but neither
    references any context. That comes later.


.. code-block:: md

   ## greet
   * greet
       - utter_greet

   ## happy
   * thankyou
       - utter_youarewelcome
   ...

   ## compare_reviews_venues
   * compare_reviews
       - action_show_venue_reviews

   ## compare_reviews_concerts
   * compare_reviews
       - action_show_concert_reviews


Interactive Learning
^^^^^^^^^^^^^^^^^^^^

Run the script ``train_online.py``. This first creats a stateless policy by combining the stories
we've provided into longer dialogues, and then trains the policy on that
dataset.

It then runs the bot so that you can provide feedback to train it (this is
where the learning *becomes interactive*):

**Happy paths**

.. note::

    We haven't connected an NLU tool here,
    so when you type messages to the bot you have to
    type the intent starting with a ``/`` (see :ref:`fixed_intent_format`).
    If you want to use Rasa NLU / wit.ai / Lex you
    can just swap the ``Interpreter`` class in ``run.py`` and ``train_online.py``.

We now start talking to the bot by directly entering the intents. For
example, if we type ``/greet``, we get the following prompt:

.. code-block:: text

   /greet
   ------
   Chat history:

        bot did:    None
        bot did:	action_listen
        user said:	/greet

                whose intent is:	greet

   we currently have slots: concerts: None, venues: None
   ------
   The bot wants to [utter_greet] due to the intent. Is this correct?

       1.	Yes
       2.	No, intent is right but the action is wrong
       3.	The intent is wrong
       0.	Export current conversations as stories and quit


This gives you all the info you should hopefully need to decide
what the bot *should* have done. In this case, the bot chose the right
action ('utter_greet'), so we type ``1`` and hit enter.
Then we type ``1`` again, because 'action_listen' is the correct action after greeting.
We continue this loop until the bot chooses the wrong action.

**Providing feedback on errors**

If you ask ``/search_concerts``, the bot should suggest ``action_search_concerts`` and then ``action_listen``.
Now let's ask it to ``/compare_reviews``. The bot happens to choose the wrong one out of the two
possibilities we wrote in the stories:

.. code-block:: text

   /compare_reviews
   ------
   Chat history:

        bot did:	action_search_concerts
        bot did:	action_listen
        user said:	/compare_reviews

        	   whose intent is:	compare_reviews

   we currently have slots: concerts: [{'artist': 'Foo Fighters', 'reviews': 4.5}, {'artist': 'Katy Perry', 'reviews': 5.0}], venues: None
   ------
   The bot wants to [action_show_venue_reviews] due to the intent. Is this correct?

       1.	Yes
       2.	No, intent is right but the action is wrong
       3.	The intent is wrong
       0.	Export current conversations as stories and quit


Now we type ``2``, because it chose the wrong action, and we get a new
prompt asking for the correct one. This also shows the probabilities the
model has assigned to each of the actions.

.. code-block:: text

   what is the next action for the bot?

        0                           action_listen    0.19
        1                          action_restart    0.00
        2                           utter_default    0.00
        3                             utter_greet    0.03
        4                           utter_goodbye    0.03
        5                     utter_youarewelcome    0.02
        6                  action_search_concerts    0.09
        7                    action_search_venues    0.02
        8             action_show_concert_reviews    0.29
        9               action_show_venue_reviews    0.33



In this case, the bot should ``action_show_concert_reviews`` (rather than venue
reviews!) so we type ``8`` and hit enter.

.. note::

    The policy model will get updated *on-the-fly*,
    so that it's less likely to make the same mistake again.
    You can also export all of the conversations you have with the bot so
    you can add these as training stories in the future.

Now we can keep talking to the bot for as long as we like to create a longer
conversation. At any point you can type ``0`` and the bot will write the
current conversation to a file and exit the conversation. Make sure to
combine the dumped story with your original training data for the next
training.

.. note::

    If you run the bot with not enough training data, it might get ``action_listen``
    as a most probable response to your input and therefore do nothing.
    If you continue to input something and get no answer, please head to
    interactive training and check if ``action_listen`` was chosen as a response.
    Correct the bot's behaviour, add additional stories and run ``train.py`` then
    run the bot again.

Motivation: Why Interactive Learning?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

There are some complications to chatbot training which makes it more
tricky than most machine learning problems.

The first is that there are several ways of getting to the same goal, and
they may all be equivalently good. Therefore it is wrong to say with
certainty that given X, you should do Y, and if you do not do exactly Y then
you are wrong. This is essentially what you do in a fully supervised
learning case. We want the bot to be able to learn it can get to a
successful state through a number of different means.

Secondly, the utterances from users will be strongly affected by the
actions of the bot. That means that a network trained on pre-collected
data will suffer from `exposure bias <https://arxiv.org/abs/1511.06732>`_.
This is when a system is trained to make predictions but is never given the
ability to train on its own predictions, instead being given the
ground truth every time. This has been shown to have issues when trying
to predict sequences of multiple steps into the future.

Furthermore, from a practical perspective, Rasa Core developers should be
able to train via the `Wizard of Oz <https://en.wikipedia
.org/wiki/Wizard_of_Oz_experiment>`_ method. This means that if you want a
bot to do a certain task, you can simply pretend to be a bot for a little
while and at the end it will learn how to respond. This is a good way of
learning how to make the conversation natural and flowing.

