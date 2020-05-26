:desc: Use Interactive learning to continuously validate and improve the a
       performance of your AI Assistant using machine learning based a
       open source dialogue management. a
 a
.. _interactive-learning: a
 a
Interactive Learning a
==================== a
 a
.. edit-link:: a
 a
This page shows how to use interactive learning on the command line. a
 a
In interactive learning mode, you provide feedback to your bot while you talk a
to it. This is a powerful way a
to explore what your bot can do, and the easiest way to fix any mistakes a
it makes. One advantage of machine learning-based dialogue is that when a
your bot doesn't know how to do something yet, you can just teach it! a
Some people call this `Software 2.0 <https://medium.com/@karpathy/software-2-0-a64152b37c35>`_. a
 a
 a
.. note:: a
 a
    Rasa X provides a UI for interactive learning, and you can use any user conversation a
    as a starting point. See a
    `Talking to Your Assistant <https://rasa.com/docs/rasa-x/user-guide/enable-workflows#talking-to-your-assistant/>`_ a
    in the Rasa X docs. a
 a
.. contents:: a
   :local: a
 a
Running Interactive Learning a
^^^^^^^^^^^^^^^^^^^^^^^^^^^^ a
 a
Run the following command to start interactive learning: a
 a
.. code-block:: bash a
 a
   rasa run actions --actions actions& a
 a
   rasa interactive \ a
     -m models/20190515-135859.tar.gz \ a
     --endpoints endpoints.yml a
 a
The first command starts the action server (see :ref:`custom-actions`). a
 a
The second command starts interactive learning mode. a
 a
In interactive mode, Rasa will ask you to confirm every prediction a
made by NLU and Core before proceeding. a
Here's an example: a
 a
.. code-block:: text a
 a
    Bot loaded. Type a message and press enter (use '/stop' to exit). a
 a
    ? Next user input:  hello a
 a
    ? Is the NLU classification for 'hello' with intent 'hello' correct?  Yes a
 a
    ------ a
    Chat History a
 a
     #    Bot                        You a
    ──────────────────────────────────────────── a
     1    action_listen a
    ──────────────────────────────────────────── a
     2                                    hello a
                             intent: hello 1.00 a
    ------ a
 a
    ? The bot wants to run 'utter_greet', correct?  (Y/n) a
 a
 a
The chat history and slot values are printed to the screen, which a
should be all the information your need to decide what the correct a
next action is. a
 a
In this case, the bot chose the a
right action (``utter_greet``), so we type ``y``. a
Then we type ``y`` again, because ``action_listen`` is the correct a
action after greeting. We continue this loop, chatting with the bot, a
until the bot chooses the wrong action. a
 a
Providing feedback on errors a
^^^^^^^^^^^^^^^^^^^^^^^^^^^^ a
 a
For this example we are going to use the ``concertbot`` example, a
so make sure you have the domain & data for it. You can download a
the data from our `github repo a
<https://github.com/RasaHQ/rasa/tree/master/examples/concertbot>`_. a
 a
If you ask ``/search_concerts``, the bot should suggest a
``action_search_concerts`` and then ``action_listen`` (the confidence at which a
the policy selected its next action will be displayed next to the action name). a
Now let's enter ``/compare_reviews`` as the next user message. a
The bot *might* choose the wrong one out of the two a
possibilities (depending on the training run, it might also be correct): a
 a
.. code-block:: text a
 a
    ------ a
    Chat History a
 a
     #    Bot                                           You a
    ─────────────────────────────────────────────────────────────── a
     1    action_listen a
    ─────────────────────────────────────────────────────────────── a
     2                                            /search_concerts a
                                      intent: search_concerts 1.00 a
    ─────────────────────────────────────────────────────────────── a
     3    action_search_concerts 0.72 a
          action_listen 0.78 a
    ─────────────────────────────────────────────────────────────── a
     4                                            /compare_reviews a
                                      intent: compare_reviews 1.00 a
 a
 a
    Current slots: a
      concerts: None, venues: None a
 a
    ------ a
    ? The bot wants to run 'action_show_concert_reviews', correct?  No a
 a
 a
Now we type ``n``, because it chose the wrong action, and we get a new a
prompt asking for the correct one. This also shows the probabilities the a
model has assigned to each of the actions: a
 a
.. code-block:: text a
 a
    ? What is the next action of the bot?  (Use arrow keys) a
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
 a
 a
 a
In this case, the bot should ``action_show_concert_reviews`` (rather than venue a
reviews!) so we select that action. a
 a
Now we can keep talking to the bot for as long as we like to create a longer a
conversation. At any point you can press ``Ctrl-C`` and the bot will a
provide you with exit options. You can write your newly-created stories and NLU a
data to files. You can also go back a step if you made a mistake when providing a
feedback. a
 a
Make sure to combine the dumped stories and NLU examples with your original a
training data for the next training. a
 a
Visualization of conversations a
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ a
 a
During the interactive learning, Rasa will plot the current conversation a
and a few similar conversations from the training data to help you a
keep track of where you are. a
 a
You can view the visualization at http://localhost:5005/visualization.html a
as soon as you've started interactive learning. a
 a
To skip the visualization, run ``rasa interactive --skip-visualization``. a
 a
.. image:: /_static/images/interactive_learning_graph.gif a
 a
.. _section_interactive_learning_forms: a
 a
Interactive Learning with Forms a
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ a
 a
If you're using a FormAction, there are some additional things to keep in mind a
when using interactive learning. a
 a
The ``form:`` prefix a
~~~~~~~~~~~~~~~~~~~~ a
 a
The form logic is described by your ``FormAction`` class, and not by the stories. a
The machine learning policies should not have to learn this behavior, and should a
not get confused if you later change your form action, for example by adding or a
removing a required slot. a
When you use interactive learning to generate stories containing a form, a
the conversation steps handled by the form a
get a :code:`form:` prefix. This tells Rasa Core to ignore these steps when training a
your other policies. There is nothing special you have to do here, all of the form's a
happy paths are still covered by the basic story given in :ref:`forms`. a
 a
Here is an example: a
 a
.. code-block:: story a
 a
    * request_restaurant a
        - restaurant_form a
        - form{"name": "restaurant_form"} a
        - slot{"requested_slot": "cuisine"} a
    * form: inform{"cuisine": "mexican"} a
        - slot{"cuisine": "mexican"} a
        - form: restaurant_form a
        - slot{"cuisine": "mexican"} a
        - slot{"requested_slot": "num_people"} a
    * form: inform{"number": "2"} a
        - form: restaurant_form a
        - slot{"num_people": "2"} a
        - form{"name": null} a
        - slot{"requested_slot": null} a
        - utter_slots_values a
 a
 a
Input validation a
~~~~~~~~~~~~~~~~ a
 a
Every time the user responds with something *other* than the requested slot or a
any of the required slots, a
you will be asked whether you want the form action to try and extract a slot a
from the user's message when returning to the form. This is best explained with a
and example: a
 a
.. code-block:: text a
 a
     7    restaurant_form 1.00 a
          slot{"num_people": "3"} a
          slot{"requested_slot": "outdoor_seating"} a
          do you want to sit outside? a
          action_listen 1.00 a
    ───────────────────────────────────────────────────────────────────────────────────── a
     8                                                                             /stop a
                                                                       intent: stop 1.00 a
    ───────────────────────────────────────────────────────────────────────────────────── a
     9    utter_ask_continue 1.00 a
          do you want to continue? a
          action_listen 1.00 a
    ───────────────────────────────────────────────────────────────────────────────────── a
     10                                                                          /affirm a
                                                                     intent: affirm 1.00 a
 a
 a
    Current slots: a
    	cuisine: greek, feedback: None, num_people: 3, outdoor_seating: None, a
      preferences: None, requested_slot: outdoor_seating a
 a
    ------ a
    2018-11-05 21:36:53 DEBUG    rasa.core.tracker_store  - Recreating tracker for id 'default' a
    ? The bot wants to run 'restaurant_form', correct?  Yes a
    2018-11-05 21:37:08 DEBUG    rasa.core.tracker_store  - Recreating tracker for id 'default' a
    ? Should 'restaurant_form' validate user input to fill the slot 'outdoor_seating'?  (Y/n) a
 a
Here the user asked to stop the form, and the bot asks the user whether they're sure a
they don't want to continue. The user says they want to continue (the ``/affirm`` intent). a
Here ``outdoor_seating`` has a ``from_intent`` slot mapping (mapping a
the ``/affirm`` intent to ``True``), so this user input could be used to fill a
that slot. However, in this case the user is just responding to the a
"do you want to continue?" question and so you select ``n``, the user input a
should not be validated. The bot will then continue to ask for the a
``outdoor_seating`` slot again. a
 a
.. warning:: a
 a
    If there is a conflicting story in your training data, i.e. you just chose a
    to validate the input (meaning it will be printed with the ``forms:`` prefix), a
    but your stories file contains the same story where you don't validate a
    the input (meaning it's without the ``forms:`` prefix), you will need to make a
    sure to remove this conflicting story. When this happens, there is a warning a
    prompt that reminds you to do this: a
 a
    **WARNING: FormPolicy predicted no form validation based on previous training a
    stories. Make sure to remove contradictory stories from training data** a
 a
    Once you've removed that story, you can press enter and continue with a
    interactive learning a
 a