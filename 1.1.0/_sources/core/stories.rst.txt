:desc: Stories are used to teach Rasa Stack real conversation designs to learn
       from providing the basis for a scalable machine learning dialogue management.

.. _stories:

Stories
=======

A training example for the Rasa Core dialogue system is called a **story**.
This is a guide to the story data format.

.. note::

   You can also **spread your stories across multiple files** and specify the
   folder containing the files for most of the scripts (e.g. training,
   visualization). The stories will be treated as if they would have
   been part of one large file.


Format
------

Here's an example from the `bAbI <https://research.fb.com/downloads/babi/>`_
dataset (converted into Rasa stories):

.. code-block:: story

   ## story_07715946    <!-- name of the story - just for debugging -->
   * greet
      - action_ask_howcanhelp
   * inform{"location": "rome", "price": "cheap"}  <!-- user utterance, in format intent{entities} -->
      - action_on_it
      - action_ask_cuisine
   * inform{"cuisine": "spanish"}
      - action_ask_numpeople        <!-- action that the bot should execute -->
   * inform{"people": "six"}
      - action_ack_dosearch


This is what we call a **story**.


- A story starts with a name preceded by two hashes ``## story_03248462``.
  You can call the story anything you like, but it can be very useful for
  debugging to give them descriptive names!
- The end of a story is denoted by a newline, and then a new story
  starts again with ``##``.
- Messages sent by the user are shown as lines starting with ``*``
  in the format ``intent{"entity1": "value", "entity2": "value"}``.
- Actions executed by the bot are shown as lines starting with ``-``
  and contain the name of the action.
- Events returned by an action are on lines immediately after that
  action. For example, if an action returns a ``SetSlot`` event,
  this is shown as the line ``- slot{"slot_name": "value"}``


Checkpoints
-----------

You can use ``> checkpoints`` to modularize and simplify your training
data. Checkpoints can be useful, but **do not overuse them**. Using
lots of checkpoints can quickly make your example stories hard to
understand. It makes sense to use them if a story block is repeated
very often in different stories, but stories *without* checkpoints
are easier to read and write. Here is an example story file which
contains checkpoints:

.. code-block:: story

    ## first story
    * hello
       - action_ask_user_question
    > check_asked_question

    ## user affirms question
    > check_asked_question
    * affirm
      - action_handle_affirmation

    ## user denies question
    > check_asked_question
    * deny
      - action_handle_denial


OR Statements
-------------

Another way to write shorter stories, or to handle multiple intents
the same way, is to use an ``OR`` statement. For example if you ask
the user to confirm something, and we want to treat the ``affirm``
and ``thankyou`` intents in the same way. The story below will be
converted into two stories at training time. Just like checkpoints,
``OR`` statements can be useful, but if you are using a lot of them,
it is probably better to restructure your domain and/or intents:

.. code-block:: story

    ## story
    ...
      - utter_ask_confirm
    * affirm OR thankyou
      - action_handle_affirmation


.. note::

   Adding lines to your stories with many ``OR`` statements
   will slow down training.
