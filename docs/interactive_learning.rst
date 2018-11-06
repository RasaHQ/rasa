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


Load up an existing bot
^^^^^^^^^^^^^^^^^^^^^^^

We have a basic working bot, and want to teach it by providing
feedback on mistakes it makes.

Run the following to start interactive learning:

.. code-block:: bash

   python -m rasa_core_sdk.endpoint --actions actions&

   python -m rasa_core.train \
     --interactive -o models/dialogue \
     -d domain.yml -s stories.md \
     --endpoints endpoints.yml

To include an existing model to identify intents use --nlu models/current/nlu in the above command. Else interactive learning will use a default REGEX to intentify default intents from the user input text.

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

Providing feedback on errors
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

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

Form Action corrections
^^^^^^^^^^^^^^^^^^^^^^^

If you're using a FormAction, there are some additional things to note
apart from the standard interactive learning behaviour described above.

The ``form:`` prefix
~~~~~~~~~~~~~~~~~~~~

Since forms work mainly with rule based behaviour, Rasa Core should not be
trained on anything that was predicted within a form. To still capture what's
going in the form action during interactive learning, these events will have
a ``form:`` prefix in the generated story. For example:

.. code-block:: story

    * request_restaurant
        - restaurant_form
        - form{"name": "restaurant_form"}
        - slot{"requested_slot": "cuisine"}
    * form: inform{"cuisine": "mexican"}
        - slot{"cuisine": "mexican"}
        - form: restaurant_form
        - slot{"cuisine": "mexican"}
        - slot{"requested_slot": "num_people"}
    * form: inform{"number": "2"}
        - form: restaurant_form
        - slot{"num_people": "2"}
        - form{"name": null}
        - slot{"requested_slot": null}
        - utter_slots_values

Input validation
~~~~~~~~~~~~~~~~

Every time the user enters some unexpected input (i.e. doesn't provide the
requested slot), you will be asked whether you want the form action to validate
the users input when returning to it. This is necessary because in certain
situations you may not want the users current input to be used to fill the
current requested slot. Take the example below:

.. code-block:: text

     7    restaurant_form 1.00
          slot{"num_people": "3"}
          slot{"requested_slot": "outdoor_seating"}
          do you want to sit outside?
          action_listen 1.00
    ─────────────────────────────────────────────────────────────────────────────────────
     8                                                                             /stop
                                                                       intent: stop 1.00
    ─────────────────────────────────────────────────────────────────────────────────────
     9    utter_ask_continue 1.00
          do you want to continue?
          action_listen 1.00
    ─────────────────────────────────────────────────────────────────────────────────────
     10                                                                          /affirm
                                                                     intent: affirm 1.00


    Current slots:
    	cuisine: greek, feedback: None, num_people: 3, outdoor_seating: None,
      preferences: None, requested_slot: outdoor_seating

    ------
    2018-11-05 21:36:53 DEBUG    rasa_core.tracker_store  - Recreating tracker for id 'default'
    ? The bot wants to run 'restaurant_form', correct?  Yes
    2018-11-05 21:37:08 DEBUG    rasa_core.tracker_store  - Recreating tracker for id 'default'
    ? Should 'restaurant_form' validate user input to fill the slot 'outdoor_seating'?  (Y/n)

Here the user asked to stop the form, and the bot asks the user whether he's sure
he doesn't want to continue. The user says he wants to continue with
``/affirm``. Here ``outdoor_seating`` has a ``from_intent`` slot mapping (mapping
the ``/affirm`` intent to ``True``), so this user input could be used to fill
that slot. However, in this case the user is just responding to the
"do you want to continue?" question and so you select ``n``, the user input
should not be validated. The bot will then continue to ask for the
``outdoor_seating`` slot again.

.. warning::

    If there is a conflicting story in your training data, i.e. you just chose
    to validate the input (meaning it will be printed with the ``forms:`` prefix),
    but your stories file contains the same story where you don't validate
    the input (meaning it's without the ``forms:`` prefix), you will need to make
    sure to remove this conflicting story. When this happens, there is a warning
    prompt that reminds you to do this:

    **WARNING: FormPolicy predicted no form validation based on previous training
    stories. Make sure to remove contradictory stories from training data**
    
    Once you've removed that story, you can press enter and continue with
    interactive learning


Visualization of conversations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

During the interactive learning, Core will plot the current conversation
and close surrounding conversations from the training data to help you
keep track of where you are.

You can view the visualization at http://localhost:5005/visualization.html
as soon as you have started the interactive learning.

To skip the visualization, pass ``--skip_visualization`` to the training
script.

.. image:: _static/images/interactive_learning_graph.gif

.. include:: feedback.inc
