:desc: Use Interactive learning to continuously validate and improve the
       performance of your AI Assistant using machine learning based
       open source dialogue management.

.. _interactive-learning:

Interactive Learning
====================

.. edit-link::

This page shows how to use interactive learning on the command line.

In interactive learning mode, you provide feedback to your bot while you talk
to it. This is a powerful way
to explore what your bot can do, and the easiest way to fix any mistakes
it makes. One advantage of machine learning-based dialogue is that when
your bot doesn't know how to do something yet, you can just teach it!
Some people call this `Software 2.0 <https://medium.com/@karpathy/software-2-0-a64152b37c35>`_.


.. note::

    Rasa X provides a UI for interactive learning, and you can use any user conversation
    as a starting point. See `Annotate Conversations <https://rasa.com/docs/rasa-x/annotate-conversations/>`_ in the Rasa X docs.

.. contents::
   :local:

Running Interactive Learning
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Run the following command to start interactive learning:

.. code-block:: bash

   rasa run actions --actions actions&

   rasa interactive \
     -m models/20190515-135859.tar.gz \
     --endpoints endpoints.yml

The first command starts the action server (see :ref:`custom-actions`).

The second command starts interactive learning mode.

In interactive mode, Rasa will ask you to confirm every prediction
made by NLU and Core before proceeding.
Here's an example:

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


The chat history and slot values are printed to the screen, which
should be all the information your need to decide what the correct
next action is.

In this case, the bot chose the
right action (``utter_greet``), so we type ``y``.
Then we type ``y`` again, because ``action_listen`` is the correct
action after greeting. We continue this loop, chatting with the bot,
until the bot chooses the wrong action.

Providing feedback on errors
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For this example we are going to use the ``concertbot`` example,
so make sure you have the domain & data for it. You can download
the data from our `github repo
<https://github.com/RasaHQ/rasa/tree/master/examples/concertbot>`_.

If you ask ``/search_concerts``, the bot should suggest
``action_search_concerts`` and then ``action_listen`` (the confidence at which
the policy selected its next action will be displayed next to the action name).
Now let's enter ``/compare_reviews`` as the next user message.
The bot *might* choose the wrong one out of the two
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
provide you with exit options. You can write your newly-created stories and NLU
data to files. You can also go back a step if you made a mistake when providing
feedback.

Make sure to combine the dumped stories and NLU examples with your original
training data for the next training.

Visualization of conversations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

During the interactive learning, Rasa will plot the current conversation
and a few similar conversations from the training data to help you
keep track of where you are.

You can view the visualization at http://localhost:5005/visualization.html
as soon as you've started interactive learning.

To skip the visualization, run ``rasa interactive --skip-visualization``.

.. image:: /_static/images/interactive_learning_graph.gif

.. _section_interactive_learning_forms:

Interactive Learning with Forms
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you're using a FormAction, there are some additional things to keep in mind
when using interactive learning.

The ``form:`` prefix
~~~~~~~~~~~~~~~~~~~~

The form logic is described by your ``FormAction`` class, and not by the stories.
The machine learning policies should not have to learn this behavior, and should
not get confused if you later change your form action, for example by adding or
removing a required slot.
When you use interactive learning to generate stories containing a form,
the conversation steps handled by the form
get a :code:`form:` prefix. This tells Rasa Core to ignore these steps when training
your other policies. There is nothing special you have to do here, all of the form's
happy paths are still covered by the basic story given in :ref:`forms`.

Here is an example:

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

Every time the user responds with something *other* than the requested slot or
any of the required slots,
you will be asked whether you want the form action to try and extract a slot
from the user's message when returning to the form. This is best explained with
and example:

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
    2018-11-05 21:36:53 DEBUG    rasa.core.tracker_store  - Recreating tracker for id 'default'
    ? The bot wants to run 'restaurant_form', correct?  Yes
    2018-11-05 21:37:08 DEBUG    rasa.core.tracker_store  - Recreating tracker for id 'default'
    ? Should 'restaurant_form' validate user input to fill the slot 'outdoor_seating'?  (Y/n)

Here the user asked to stop the form, and the bot asks the user whether they're sure
they don't want to continue. The user says they want to continue (the ``/affirm`` intent).
Here ``outdoor_seating`` has a ``from_intent`` slot mapping (mapping
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
