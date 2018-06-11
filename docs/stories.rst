.. _stories:

Story Data Format
=================

A training data sample for the dialogue system is called a **story**. This
shows you how to define them and how to visualise them.

Format
------

Here's an example from the bAbi data:

.. code-block:: md

   ## story_07715946                     <!-- name of the story - just for debugging -->
   * greet
      - action_ask_howcanhelp
   * inform{"location": "rome", "price": "cheap"}  <!-- user utterance, in format intent{entities} -->
      - action_on_it                     
      - action_ask_cuisine
   * inform{"cuisine": "spanish"}
      - action_ask_numpeople             <!-- action of the bot to execute -->
   * inform{"people": "six"}
      - action_ack_dosearch


This is what we call a **story**. A story starts with a name preceded by two
hashes ``## story_03248462``, this is arbitrary but can be used for debugging.
The end of a story is denoted by a newline, and then a new story starts again with ``##``.

You can use ``> checkpoints`` to modularize and simplify your training data:

.. code-block:: md

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

.. note::

   You can also **spread your stories across multiple files** and specify the
   folder containing the files for most of the scripts (e.g. training,
   visualization). The stories will be treated as if they would have
   been part of one large file.

