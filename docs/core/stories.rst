:desc: Stories are used to teach Rasa real conversation designs to learn
       from providing the basis for a scalable machine learning dialogue management.

.. _stories:

Stories
=======

A training example for the Rasa dialogue system is called a **story**.
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
   action. For example, if an action returns a ``SlotSet`` event,
  this is shown as the line ``- slot{"slot_name": "value"}``.

Let's now take a slightly more detailed look at each of these components,
along with some things you should keep in mind while using them to write
your own stories.

User Messages
~~~~~~~~~~~~~
While writing stories, you do not have to deal with the specific contents of
the messages that the users send. Instead, you can take advantage of the output
from the NLU pipeline, which lets you use just the combination of an intent and
entities to refer to all the possible messages the users can send to mean the
same thing.

It is important to include the entities as well because the policies learn to
predict the next action based on a *combination* of both the intent and
entities.

Actions
~~~~~~~
While writing stories, you will encounter two types of actions - utterances
and custom actions. Utterances are hardcoded messages that a bot can respond
with. Custom actions, on the other hand, involve custom code being executed. 

All actions (both utterances and custom actions) executed by the bot are shown
as lines starting with ``-`` followed by the name of the action. For custom
actions, the action name is the string you choose to return from the ``name``
method of the custom action class. The convention is that utterances begin
with the prefix ``utter_`` and custom actions begin with ``action_``.

Events
~~~~~~
Events such as setting a slot or activating/deactivating a form have to be
explicitly written out as part of the stories. Having to include the events
returned by a custom action separately, when that custom action is already
part of a story might seem redundant. However, since Rasa cannot easily
determine this fact during training, this step is a necessary evil.

In order to get around this pitfall, the recommended way to write these
stories is to use :ref:`interactive-learning`.


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
the same way, is to use an ``OR`` statement. For example, if you ask
the user to confirm something, and you want to treat the ``affirm``
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
