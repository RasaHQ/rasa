:desc: Interactive Learning with Rasa Core

.. _interactive_learning:

Interactive Learning
====================


Interactive learning means giving feedback to your bot while you talk to it. It is a powerful tool!
Interactive learning is a powerful way to explore what your bot can do, and the easiest
way to fix any mistakes it makes. One advantage of machine learning based dialogue is that
when your bot doesn't know how to do something yet, you can just teach it! Some people
are calling this `Software 2.0 <https://tesla.com>`_.


1. Load up an existing bot
^^^^^^^^^^^^^^^^^^^^^^^^^^

We have a basic working bot, and want to teach it by providing feedback on mistakes it makes. 

Run the following to start interactive learning:

.. code-block:: bash

   python -m rasa_core_sdk.endpoint --actions actions&
   python -m rasa_core.run --interactive -d models/dialogue --stories-out stories_interactive.md -u models/default/nlu

The first command starts the action server (see :ref:`customactions`).

The second command starts the bot in interactive mode.
In interactive mode, the bot will ask you to confirm it has chosen the right action before proceeding.


.. code-block:: text

   hello
   ------
   Chat history:

        bot did:    None
        bot did:	action_listen
        user said:	hello

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

   which has better reviews?
   ------
   Chat history:

        bot did:	action_search_concerts
        bot did:	action_listen
        user said:      which has better reviews?

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


Now we can keep talking to the bot for as long as we like to create a longer
conversation. At any point you can type ``0`` and the bot will write the
current conversation to a file and exit the conversation. Make sure to
combine the dumped story with your original training data for the next
training.

