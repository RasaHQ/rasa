:desc: Stories are used to teach Rasa real conversation designs to learn
       from providing the basis for a scalable machine learning dialogue management.

.. _stories:

Stories
=======

.. edit-link::

.. contents::
   :local:

Rasa stories are a form of training data used to train the Rasa's dialogue management models.

A story is a representation of a conversation between a user and an AI assistant, converted into a specific format where user inputs are expressed as corresponding intents (and entities where necessary) while the responses of an assistant are expressed as corresponding action names.

A training example for the Rasa Core dialogue system is called a **story**.
This is a guide to the story data format.

.. note::
   You can also **spread your stories across multiple files** and specify the
   folder containing the files for most of the scripts (e.g. training,
   visualization). The stories will be treated as if they would have
   been part of one large file.


Format
------

Here's an example of a dialogue in the Rasa story format:

.. code-block:: story

   ## greet + location/price + cuisine + num people    <!-- name of the story - just for debugging -->
   * greet
      - action_ask_howcanhelp
   * inform{"location": "rome", "price": "cheap"}  <!-- user utterance, in format intent{entities} -->
      - action_on_it
      - action_ask_cuisine
   * inform{"cuisine": "spanish"}
      - action_ask_numpeople        <!-- action that the bot should execute -->
   * inform{"people": "six"}
      - action_ack_dosearch


What makes up a story?
~~~~~~~~~~~~~~~~~~~~~~

- A story starts with a name preceded by two hashes ``## story_03248462``.
  You can call the story anything you like, but it can be very useful for
  debugging to give them descriptive names!
- The end of a story is denoted by a newline, and then a new story
  starts again with ``##``.
- Messages sent by the user are shown as lines starting with ``*``
  in the format ``intent{"entity1": "value", "entity2": "value"}``.
- Actions executed by the bot are shown as lines starting with ``-``
  and contain the name of the action.
- Events returned by an action are on lines immediately after that action.
  For example, if an action returns a ``SlotSet`` event, this is shown as
  ``slot{"slot_name": "value"}``.


User Messages
~~~~~~~~~~~~~
While writing stories, you do not have to deal with the specific contents of
the messages that the users send. Instead, you can take advantage of the output
from the NLU pipeline, which lets you use just the combination of an intent and
entities to refer to all the possible messages the users can send to mean the
same thing.

It is important to include the entities here as well because the policies learn
to predict the next action based on a *combination* of both the intent and
entities (you can, however, change this behavior using the
:ref:`use_entities <use_entities>` attribute).

.. warning::
    ``/`` symbol is reserved as a delimiter to separate retrieval intents from response text identifiers.
    Refer to ``Training Data Format`` section of :ref:`retrieval-actions` for more details on this format.
    If any of the intent names contain the delimiter, the file containing these stories will be considered as a training
    file for :ref:`response-selector` model and will be ignored for training Core models.

Actions
~~~~~~~
While writing stories, you will encounter two types of actions: utterance actions
and custom actions. Utterance actions are hardcoded messages that a bot can respond
with. Custom actions, on the other hand, involve custom code being executed.

All actions (both utterance actions and custom actions) executed by the bot are shown
as lines starting with ``-`` followed by the name of the action.

The responses for utterance actions must begin with the prefix ``utter_``, and must match the name
of the response defined in the domain.

For custom actions, the action name is the string you choose to return from
the ``name`` method of the custom action class. Although there is no restriction
on naming your custom actions (unlike utterance actions), the best practice here is to
prefix the name with ``action_``.

Events
~~~~~~
Events such as setting a slot or activating/deactivating a form have to be
explicitly written out as part of the stories. Having to include the events
returned by a custom action separately, when that custom action is already
part of a story might seem redundant. However, since Rasa cannot
determine this fact during training, this step is necessary.

You can read more about events :ref:`here <events>`.

Slot Events
***********
Slot events are written as ``- slot{"slot_name": "value"}``. If this slot is set
inside a custom action, it is written on the line immediately following the
custom action event. If your custom action resets a slot value to `None`, the
corresponding event for that would be ``-slot{"slot_name": null}``.

Form Events
***********
There are three kinds of events that need to be kept in mind while dealing with
forms in stories.

- A form action event (e.g. ``- restaurant_form``) is used in the beginning when first starting a form, and also while resuming the form action when the form is already active.
- A form activation event (e.g. ``- form{"name": "restaurant_form"}``) is used right after the first form action event.
- A form deactivation event (e.g. ``- form{"name": null}``), which is used to deactivate the form.


.. note::
    In order to get around the pitfall of forgetting to add events, the recommended
    way to write these stories is to use :ref:`interactive learning <interactive-learning>`.


Checkpoints and OR statements
-----------------------------

Checkpoints and OR statements should both be used with caution, if at all.
There is usually a better way to achieve what you want by using forms and/or
retrieval actions.


Checkpoints
~~~~~~~~~~~

You can use ``> checkpoints`` to modularize and simplify your training
data. Checkpoints can be useful, but **do not overuse them**. Using
lots of checkpoints can quickly make your example stories hard to
understand. It makes sense to use them if a story block is repeated
very often in different stories, but stories *without* checkpoints
are easier to read and write. Here is an example story file which
contains checkpoints (note that you can attach more than one checkpoint
at a time):

.. code-block:: story

    ## first story
    * greet
       - action_ask_user_question
    > check_asked_question

    ## user affirms question
    > check_asked_question
    * affirm
      - action_handle_affirmation
    > check_handled_affirmation

    ## user denies question
    > check_asked_question
    * deny
      - action_handle_denial
    > check_handled_denial

    ## user leaves
    > check_handled_denial
    > check_handled_affirmation
    * goodbye
      - utter_goodbye

.. note::
   Unlike regular stories, checkpoints are not restricted to starting with an
   input from the user. As long as the checkpoint is inserted at the right points
   in the main stories, the first event can be a custom action or a response action 
   as well.


OR Statements
~~~~~~~~~~~~~

Another way to write shorter stories, or to handle multiple intents
the same way, is to use an ``OR`` statement. For example, if you ask
the user to confirm something, and you want to treat the ``affirm``
and ``thankyou`` intents in the same way. The story below will be
converted into two stories at training time:


.. code-block:: story

    ## story
    ...
      - utter_ask_confirm
    * affirm OR thankyou
      - action_handle_affirmation

Just like checkpoints, ``OR`` statements can be useful, but if you are using a
lot of them, it is probably better to restructure your domain and/or intents.


.. warning::
    Overusing these features (both checkpoints and OR statements)
    will slow down training.


End-to-End Story Evaluation Format
----------------------------------

The end-to-end story format is a format that combines both NLU and Core training data
into a single file for evaluation. You can read more about it
:ref:`here <end_to_end_evaluation>`.

.. warning::
    This format is only used for end-to-end evaluation and cannot be used for training.
