:desc: Interactive Learning with Rasa Core

.. _interactive_learning:

Interactive Learning
====================


Interactive learning means giving feedback to your bot while you talk
to it. It is a powerful tool! Interactive learning is a powerful way
to explore what your bot can do, and the easiest way to fix any mistakes
it makes. One advantage of machine learning based dialogue is that when
your bot doesn't know how to do something yet, you can just teach it!
Some people are calling this `Software 2.0 <https://medium.com/@karpathy/software-2-0-a64152b37c35>`_.


1. Load up an existing bot
^^^^^^^^^^^^^^^^^^^^^^^^^^

We have a basic working bot, and want to teach it by providing
feedback on mistakes it makes.

Run the following to start interactive learning:

.. code-block:: bash

   python -m rasa_core_sdk.endpoint --actions actions&

   python -m rasa_core.train \
     --interactive -o models/dialogue \
     -d domain.yml -s stories.md \
     --endpoints endpoints.yml

The first command starts the action server (see :ref:`customactions`).

The second command starts the bot in interactive mode.
In interactive mode, the bot will ask you to confirm it has chosen
the right action before proceeding:


.. code-block:: text

    Bot loaded. Type a message and press enter (use '/stop' to exit).

    ? Next user input:  hello

    ? Is the NLU classification for 'hello' with intent 'hello' correct?  Yes

    ------
    Chat History

     #    Bot                        You
    ────────────────────────────────────────────
     1    action_listen
    ────────────────────────────────────────────
     2                                    hello
                             intent: hello 1.00
    ------

    ? The bot wants to run 'utter_greet', correct?  (Y/n)


This gives you all the info you should hopefully need to decide
what the bot *should* have done. In this case, the bot chose the
right action (``utter_greet``), so we type ``y``.
Then we type ``y`` again, because 'action_listen' is the correct
action after greeting. We continue this loop until the bot chooses
the wrong action.

**Providing feedback on errors**

For this example we are going to use the ``concertbot`` example,
so make sure you have the domain & data for it. You can download
the data from :gh-code:`examples/concertbot`.

If you ask ``/search_concerts``, the bot should suggest
``action_search_concerts`` and then ``action_listen`` (the confidence at which
the policy selected its next action will be displayed next to the action name).
Now let's enter ``/compare_reviews`` as the next user message.
The bot **might** choose the wrong one out of the two
possibilities (depending on the training run, it might also be correct):

.. code-block:: text

    ------
    Chat History

     #    Bot                                           You
    ───────────────────────────────────────────────────────────────
     1    action_listen
    ───────────────────────────────────────────────────────────────
     2                                            /search_concerts
                                      intent: search_concerts 1.00
    ───────────────────────────────────────────────────────────────
     3    action_search_concerts 0.72
          action_listen 0.78
    ───────────────────────────────────────────────────────────────
     4                                            /compare_reviews
                                      intent: compare_reviews 1.00


    Current slots:
      concerts: None, venues: None

    ------
    ? The bot wants to run 'action_show_concert_reviews', correct?  No


Now we type ``n``, because it chose the wrong action, and we get a new
prompt asking for the correct one. This also shows the probabilities the
model has assigned to each of the actions:

.. code-block:: text

    ? What is the next action of the bot?  (Use arrow keys)
     ❯ 0.53 action_show_venue_reviews
       0.46 action_show_concert_reviews
       0.00 utter_goodbye
       0.00 action_search_concerts
       0.00 utter_greet
       0.00 action_search_venues
       0.00 action_listen
       0.00 utter_youarewelcome
       0.00 utter_default
       0.00 action_default_fallback
       0.00 action_restart



In this case, the bot should ``action_show_concert_reviews`` (rather than venue
reviews!) so we select that action.

Now we can keep talking to the bot for as long as we like to create a longer
conversation. At any point you can press ``Ctrl-C`` and the bot will
provide you with exit options, e.g. writing the created conversations as
stories to a file. Make sure to combine the dumped story with your original
training data for the next training.


.. include:: feedback.inc  

.. raw:: html
   :file: livechat.html
