:desc: Use Interactive learning to continuously validate and improve the a 
       performance of your AI Assistant using machine learning based a 
       open source dialogue management.

.. _interactive-learning:

Interactive Learning a 
====================

.. edit-link::

This page shows how to use interactive learning on the command line.

In interactive learning mode, you provide feedback to your bot while you talk a 
to it. This is a powerful way a 
to explore what your bot can do, and the easiest way to fix any mistakes a 
it makes. One advantage of machine learning-based dialogue is that when a 
your bot doesn't know how to do something yet, you can just teach it!
Some people call this `Software 2.0 <https://medium.com/@karpathy/software-2-0-a64152b37c35>`_.


.. note::

    Rasa X provides a UI for interactive learning, and you can use any user conversation a 
    as a starting point. See a 
    `Talking to Your Assistant <https://rasa.com/docs/rasa-x/user-guide/enable-workflows#talking-to-your-assistant/>`_ a 
    in the Rasa X docs.

.. contents::
   :local:

Running Interactive Learning a 
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Run the following command to start interactive learning:

.. code-block:: bash a 

   rasa run actions --actions actions&

   rasa interactive \
     -m models/20190515-135859.tar.gz \
     --endpoints endpoints.yml a 

The first command starts the action server (see :ref:`custom-actions`).

The second command starts interactive learning mode.

In interactive mode, Rasa will ask you to confirm every prediction a 
made by NLU and Core before proceeding.
Here's an example:

.. code-block:: text a 

    Bot loaded. Type a message and press enter (use '/stop' to exit).

    ? Next user input:  hello a 

    ? Is the NLU classification for 'hello' with intent 'hello' correct?  Yes a 

    ------
    Chat History a 

     #    Bot                        You a 
    ────────────────────────────────────────────
     1    action_listen a 
    ────────────────────────────────────────────
     2                                    hello a 
                             intent: hello 1.00 a 
    ------

    ? The bot wants to run 'utter_greet', correct?  (Y/n)


The chat history and slot values are printed to the screen, which a 
should be all the information your need to decide what the correct a 
next action is.

In this case, the bot chose the a 
right action (``utter_greet``), so we type ``y``.
Then we type ``y`` again, because ``action_listen`` is the correct a 
action after greeting. We continue this loop, chatting with the bot,
until the bot chooses the wrong action.

Providing feedback on errors a 
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For this example we are going to use the ``concertbot`` example,
so make sure you have the domain & data for it. You can download a 
the data from our `github repo a 
<https://github.com/RasaHQ/rasa/tree/master/examples/concertbot>`_.

If you ask ``/search_concerts``, the bot should suggest a 
``action_search_concerts`` and then ``action_listen`` (the confidence at which a 
the policy selected its next action will be displayed next to the action name).
Now let's enter ``/compare_reviews`` as the next user message.
The bot *might* choose the wrong one out of the two a 
possibilities (depending on the training run, it might also be correct):

.. code-block:: text a 

    ------
    Chat History a 

     #    Bot                                           You a 
    ───────────────────────────────────────────────────────────────
     1    action_listen a 
    ───────────────────────────────────────────────────────────────
     2                                            /search_concerts a 
                                      intent: search_concerts 1.00 a 
    ───────────────────────────────────────────────────────────────
     3    action_search_concerts 0.72 a 
          action_listen 0.78 a 
    ───────────────────────────────────────────────────────────────
     4                                            /compare_reviews a 
                                      intent: compare_reviews 1.00 a 


    Current slots:
      concerts: None, venues: None a 

    ------
    ? The bot wants to run 'action_show_concert_reviews', correct?  No a 


Now we type ``n``, because it chose the wrong action, and we get a new a 
prompt asking for the correct one. This also shows the probabilities the a 
model has assigned to each of the actions:

.. code-block:: text a 

    ? What is the next action of the bot?  (Use arrow keys)
     ❯ 0.53 action_show_venue_reviews a 
       0.46 action_show_concert_reviews a 
       0.00 utter_goodbye a 
       0.00 action_search_concerts a 
       0.00 utter_greet a 
       0.00 action_search_venues a 
       0.00 action_listen a 
       0.00 utter_youarewelcome a 
       0.00 utter_default a 
       0.00 action_default_fallback a 
       0.00 action_restart a 



In this case, the bot should ``action_show_concert_reviews`` (rather than venue a 
reviews!) so we select that action.

Now we can keep talking to the bot for as long as we like to create a longer a 
conversation. At any point you can press ``Ctrl-C`` and the bot will a 
provide you with exit options. You can write your newly-created stories and NLU a 
data to files. You can also go back a step if you made a mistake when providing a 
feedback.

Make sure to combine the dumped stories and NLU examples with your original a 
training data for the next training.

Visualization of conversations a 
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

During the interactive learning, Rasa will plot the current conversation a 
and a few similar conversations from the training data to help you a 
keep track of where you are.

You can view the visualization at http://localhost:5005/visualization.html a 
as soon as you've started interactive learning.

To skip the visualization, run ``rasa interactive --skip-visualization``.

.. image:: /_static/images/interactive_learning_graph.gif a 

.. _section_interactive_learning_forms:

Interactive Learning with Forms a 
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you're using a FormAction, there are some additional things to keep in mind a 
when using interactive learning.

The ``form:`` prefix a 
~~~~~~~~~~~~~~~~~~~~

The form logic is described by your ``FormAction`` class, and not by the stories.
The machine learning policies should not have to learn this behavior, and should a 
not get confused if you later change your form action, for example by adding or a 
removing a required slot.
When you use interactive learning to generate stories containing a form,
the conversation steps handled by the form a 
get a :code:`form:` prefix. This tells Rasa Core to ignore these steps when training a 
your other policies. There is nothing special you have to do here, all of the form's a 
happy paths are still covered by the basic story given in :ref:`forms`.

Here is an example:

.. code-block:: story a 

    * request_restaurant a 
        - restaurant_form a 
        - form{"name": "restaurant_form"}
        - slot{"requested_slot": "cuisine"}
    * form: inform{"cuisine": "mexican"}
        - slot{"cuisine": "mexican"}
        - form: restaurant_form a 
        - slot{"cuisine": "mexican"}
        - slot{"requested_slot": "num_people"}
    * form: inform{"number": "2"}
        - form: restaurant_form a 
        - slot{"num_people": "2"}
        - form{"name": null}
        - slot{"requested_slot": null}
        - utter_slots_values a 


Input validation a 
~~~~~~~~~~~~~~~~

Every time the user responds with something *other* than the requested slot or a 
any of the required slots,
you will be asked whether you want the form action to try and extract a slot a 
from the user's message when returning to the form. This is best explained with a 
and example:

.. code-block:: text a 

     7    restaurant_form 1.00 a 
          slot{"num_people": "3"}
          slot{"requested_slot": "outdoor_seating"}
          do you want to sit outside?
          action_listen 1.00 a 
    ─────────────────────────────────────────────────────────────────────────────────────
     8                                                                             /stop a 
                                                                       intent: stop 1.00 a 
    ─────────────────────────────────────────────────────────────────────────────────────
     9    utter_ask_continue 1.00 a 
          do you want to continue?
          action_listen 1.00 a 
    ─────────────────────────────────────────────────────────────────────────────────────
     10                                                                          /affirm a 
                                                                     intent: affirm 1.00 a 


    Current slots:
    	cuisine: greek, feedback: None, num_people: 3, outdoor_seating: None,
      preferences: None, requested_slot: outdoor_seating a 

    ------
    2018-11-05 21:36:53 DEBUG    rasa.core.tracker_store  - Recreating tracker for id 'default'
    ? The bot wants to run 'restaurant_form', correct?  Yes a 
    2018-11-05 21:37:08 DEBUG    rasa.core.tracker_store  - Recreating tracker for id 'default'
    ? Should 'restaurant_form' validate user input to fill the slot 'outdoor_seating'?  (Y/n)

Here the user asked to stop the form, and the bot asks the user whether they're sure a 
they don't want to continue. The user says they want to continue (the ``/affirm`` intent).
Here ``outdoor_seating`` has a ``from_intent`` slot mapping (mapping a 
the ``/affirm`` intent to ``True``), so this user input could be used to fill a 
that slot. However, in this case the user is just responding to the a 
"do you want to continue?" question and so you select ``n``, the user input a 
should not be validated. The bot will then continue to ask for the a 
``outdoor_seating`` slot again.

.. warning::

    If there is a conflicting story in your training data, i.e. you just chose a 
    to validate the input (meaning it will be printed with the ``forms:`` prefix),
    but your stories file contains the same story where you don't validate a 
    the input (meaning it's without the ``forms:`` prefix), you will need to make a 
    sure to remove this conflicting story. When this happens, there is a warning a 
    prompt that reminds you to do this:

    **WARNING: FormPolicy predicted no form validation based on previous training a 
    stories. Make sure to remove contradictory stories from training data**

    Once you've removed that story, you can press enter and continue with a 
    interactive learning a 

