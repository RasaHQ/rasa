.. _tutorial_interactive_learning:

Interactive Learning
====================

.. note::
   This is the place to start if you have a great idea for a bot but you
   don't have any conversations to use as training data. We will assume that
   you've already thought of what intents and entities you need (check out the
   `Rasa NLU <http://nlu.rasa.ai/tutorial.html#tutorial-a-simple-restaurant-search-bot>`_
   docs if you don't know what those are).

   `Example Code on GitHub <https://github.com/RasaHQ/rasa_core/tree/master/examples/concertbot>`_


Motivation: Why Interactive Learning?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

There are some complications to chatbot training which makes them more
tricky than most machine learning problems.

The first is that there
are several ways of getting to the same goal, and they may all be equivalently good.
Therefore it is wrong to say with certainty that given X, you should do Y,
and if you do not do exactly Y then you are wrong. This is essentially
what you do in a fully supervised learning case. We want the bot
to be able to learn it can get to a successful state through a number of
different means.

Secondly, the utterances from users will be strongly affected by the
actions of the bot. That means that a network trained on pre-collected
data will suffer from `exposure bias <https://arxiv.org/abs/1511.06732>`_,
this is when a system is trained to make predictions but is never given the ability to train on its own predictions, instead being given the
ground truth every time. This has been shown to have issues when trying
to predict sequences multiple steps into the future.

Also, from a practical perspective Rasa Core developers should be able to train
via the `Wizard of Oz <https://en.wikipedia.org/wiki/Wizard_of_Oz_experiment>`_
method. I.e. if you want a bot to do a certain task, you can simply
pretend to be a bot for a little while and at the end it will learn how
to respond. This is a good way of learning how to make natural and flowing


The Bot
^^^^^^^

We will build a bot that can recommend concerts to go to.
We'll show how to implement context-aware behaviour without writing a flow chart.
For example, if our user asks the question: *which of those has better reviews?*,
our bot should know whether they want to compare *musicians* or *venues*.

Let's go!


The Domain
^^^^^^^^^^

We will keep the concert domain simple, and won't add any slots just yet.
We'll also only support these intents:
``"greet", "thankyou", "goodbye", "search_concerts", "search_venues", "compare_reviews"``
Here is the domain definition:

.. literalinclude:: ../examples/concertbot/concert_domain.yml
    :linenos:

Stateless Stories
^^^^^^^^^^^^^^^^^

We start by training a stateless model on some simple dialogues in the Rasa story format.

Below is an excerpt of the stories.

In many cases, your bot's 'conversations' are just a single turn and response:
"Hello" is always met with a greeting, "goodbye!" is always met with a sign-off,
and the correct response to "thank you" is pretty much always "you're welcome".


Notice that below, we've defined two stories, showing that
``action_show_venue_reviews`` and ``action_show_concert_reviews``
are both possible responses to the ``compare_reviews`` intent, but neither references
any context. That comes later.


.. code-block:: md

   ## greet
   * _greet
       - action_greet

   ## happy
   * _thankyou
       - action_youarewelcome
   ...

   ## compare_reviews_venues
   * _compare_reviews
       - action_show_venue_reviews

   ## compare_reviews_concerts
   * _compare_reviews
       - action_show_concert_reviews



Training
^^^^^^^^

We start by training the policy to recognise these input-output pairs independently of any context.
( You can see the definition of the ``ConcertPolicy`` class in ``policy.py``. )
Run the script ``train_init.py``.
This creates a training set of conversations by randomly combining the
stories we've provided into longer dialogues, and then trains the policy on that dataset.

Then, run the script ``run.py`` to talk to the bot.
You should be able to have a conversation similar to the one below

.. note::
    we haven't connected an NLU tool here,
    so when you type messages to the bot you have to
    type the intent starting with a `_`.
    If you want to use Rasa NLU / wit.ai / Lex you
    can just swap the `Interpreter` class in `run.py`.


.. code-block:: text

   Bot loaded. Type hello and press enter :
   _greet
   hey there!
   _search_concerts
   Here's what I found:
   Katy Perry, Foo Fighters
   _goodbye
   goodbye :(


Now we'll train the bot to use context
to respond correctly to the ``compare_reviews`` intent.


Interactive Learning
^^^^^^^^^^^^^^^^^^^^

Run the script ``train_online.py``.
This first repeats the process in the ``train_init`` script, creating
a stateless policy.

It then runs the bot so that you can provide feedback to train it:

**Happy paths**

We can start talking to the bot as before,
directly entering the intents. For example, if we type ``_greet``, we get the following prompt:

.. code-block:: text

   _greet
   ------
   Chat history:

        bot did:	action_listen

        user said:	_greet

        	   whose intent is:	greet

   we currently have slots: {'location': None}

   ------
   The bot wants to [greet] due to the intent. Is this correct?

       1.	Yes
       2.	No, intent is right but the action is wrong
       3.	The intent is wrong


This gives you all the info you should hopefully need to decide
what the bot *should* have done.
In this case, the bot chose the right action ('greet'), so we type ``1`` and hit enter.
We continue this loop until the bot chooses the wrong action.

**Providing feedback on errors**

We've just asked the bot to search for concerts, and now we're asking it to compare reviews. The bot happens to choose the wrong one out of the two possibilities we wrote in the stories:

.. code-block:: text

   _compare_reviews
   ------
   Chat history:

        bot did:	action_search_concerts

        bot did:	action_suggest

        bot did:	action_listen

        user said:	_compare_reviews

        	   whose intent is:	compare_reviews

   we currently have slots: {'location': None}

   ------
   The bot wants to [show_venue_reviews] due to the intent. Is this correct?

       1.	Yes
       2.	No, intent is right but the action is wrong
       3.	The intent is wrong


Now we type ``2``, because it chose the wrong action,
and we get a new prompt asking for the correct one.
This also shows the probabilities the model has assigned to each of the actions.

.. code-block:: text

   ------
   what is the next action for the bot?

        0	default	 0.00148131744936
        1	greet	 0.0970264300704
        2	goodbye	 0.0288009047508
        3	listen	 0.00123148341663
        6	search_cinemas	0.000627864559647
        8	search_films	0.0367559418082
        9	suggest		0.0261212754995
        11	youarewelcome	0.594935178757
        13	explain_options	0.0516758263111
        14	store_slot	0.00145904591773
        15	show_cinema_reviews	0.00887114647776
        16	show_film_reviews	0.0870243906975


In this case, the bot should ``show_film_reviews`` (rather than cinema reviews!) so we type ``16`` and hit enter.

.. note:: The policy model will get updated *on-the-fly*,
   so that it's less likely to make the same mistake again.
   You can also export all of the conversations you have with the bot so you can add these as training stories in the future.

Now we can keep talking to the bot for as long as we like
to create a longer conversation. At any point you can type ``_export``
and the bot will write the current conversation to a file,
which you can then add as a training example for the future.
