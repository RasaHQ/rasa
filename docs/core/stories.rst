:desc: Stories are used to teach Rasa real conversation designs to learn a
       from providing the basis for a scalable machine learning dialogue management. a
 a
.. _stories: a
 a
Stories a
======= a
 a
.. edit-link:: a
 a
.. contents:: a
   :local: a
 a
Rasa stories are a form of training data used to train the Rasa's dialogue management models. a
 a
A story is a representation of a conversation between a user and an AI assistant, converted into a specific format where user inputs are expressed as corresponding intents (and entities where necessary) while the responses of an assistant are expressed as corresponding action names. a
 a
A training example for the Rasa Core dialogue system is called a **story**. a
This is a guide to the story data format. a
 a
.. note:: a
   You can also **spread your stories across multiple files** and specify the a
   folder containing the files for most of the scripts (e.g. training, a
   visualization). The stories will be treated as if they would have a
   been part of one large file. a
 a
 a
Format a
------ a
 a
Here's an example of a dialogue in the Rasa story format: a
 a
.. code-block:: story a
 a
   ## greet + location/price + cuisine + num people    <!-- name of the story - just for debugging --> a
   * greet a
      - action_ask_howcanhelp a
   * inform{"location": "rome", "price": "cheap"}  <!-- user utterance, in format intent{entities} --> a
      - action_on_it a
      - action_ask_cuisine a
   * inform{"cuisine": "spanish"} a
      - action_ask_numpeople        <!-- action that the bot should execute --> a
   * inform{"people": "six"} a
      - action_ack_dosearch a
 a
 a
What makes up a story? a
~~~~~~~~~~~~~~~~~~~~~~ a
 a
- A story starts with a name preceded by two hashes ``## story_03248462``. a
  You can call the story anything you like, but it can be very useful for a
  debugging to give them descriptive names! a
- The end of a story is denoted by a newline, and then a new story a
  starts again with ``##``. a
- Messages sent by the user are shown as lines starting with ``*`` a
  in the format ``intent{"entity1": "value", "entity2": "value"}``. a
- Actions executed by the bot are shown as lines starting with ``-`` a
  and contain the name of the action. a
- Events returned by an action are on lines immediately after that action. a
  For example, if an action returns a ``SlotSet`` event, this is shown as a
  ``slot{"slot_name": "value"}``. a
 a
 a
User Messages a
~~~~~~~~~~~~~ a
While writing stories, you do not have to deal with the specific contents of a
the messages that the users send. Instead, you can take advantage of the output a
from the NLU pipeline, which lets you use just the combination of an intent and a
entities to refer to all the possible messages the users can send to mean the a
same thing. a
 a
It is important to include the entities here as well because the policies learn a
to predict the next action based on a *combination* of both the intent and a
entities (you can, however, change this behavior using the a
:ref:`use_entities <use_entities>` attribute). a
 a
.. warning:: a
    ``/`` symbol is reserved as a delimiter to separate retrieval intents from response text identifiers. a
    Refer to ``Training Data Format`` section of :ref:`retrieval-actions` for more details on this format. a
    If any of the intent names contain the delimiter, the file containing these stories will be considered as a training a
    file for :ref:`response-selector` model and will be ignored for training Core models. a
 a
Actions a
~~~~~~~ a
While writing stories, you will encounter two types of actions: utterance actions a
and custom actions. Utterance actions are hardcoded messages that a bot can respond a
with. Custom actions, on the other hand, involve custom code being executed. a
 a
All actions (both utterance actions and custom actions) executed by the bot are shown a
as lines starting with ``-`` followed by the name of the action. a
 a
The responses for utterance actions must begin with the prefix ``utter_``, and must match the name a
of the response defined in the domain. a
 a
For custom actions, the action name is the string you choose to return from a
the ``name`` method of the custom action class. Although there is no restriction a
on naming your custom actions (unlike utterance actions), the best practice here is to a
prefix the name with ``action_``. a
 a
Events a
~~~~~~ a
Events such as setting a slot or activating/deactivating a form have to be a
explicitly written out as part of the stories. Having to include the events a
returned by a custom action separately, when that custom action is already a
part of a story might seem redundant. However, since Rasa cannot a
determine this fact during training, this step is necessary. a
 a
You can read more about events :ref:`here <events>`. a
 a
Slot Events a
*********** a
Slot events are written as ``- slot{"slot_name": "value"}``. If this slot is set a
inside a custom action, it is written on the line immediately following the a
custom action event. If your custom action resets a slot value to `None`, the a
corresponding event for that would be ``-slot{"slot_name": null}``. a
 a
Form Events a
*********** a
There are three kinds of events that need to be kept in mind while dealing with a
forms in stories. a
 a
- A form action event (e.g. ``- restaurant_form``) is used in the beginning when first starting a form, and also while resuming the form action when the form is already active. a
- A form activation event (e.g. ``- form{"name": "restaurant_form"}``) is used right after the first form action event. a
- A form deactivation event (e.g. ``- form{"name": null}``), which is used to deactivate the form. a
 a
 a
.. note:: a
    In order to get around the pitfall of forgetting to add events, the recommended a
    way to write these stories is to use :ref:`interactive learning <interactive-learning>`. a
 a
 a
Checkpoints and OR statements a
----------------------------- a
 a
Checkpoints and OR statements should both be used with caution, if at all. a
There is usually a better way to achieve what you want by using forms and/or a
retrieval actions. a
 a
 a
Checkpoints a
~~~~~~~~~~~ a
 a
You can use ``> checkpoints`` to modularize and simplify your training a
data. Checkpoints can be useful, but **do not overuse them**. Using a
lots of checkpoints can quickly make your example stories hard to a
understand. It makes sense to use them if a story block is repeated a
very often in different stories, but stories *without* checkpoints a
are easier to read and write. Here is an example story file which a
contains checkpoints (note that you can attach more than one checkpoint a
at a time): a
 a
.. code-block:: story a
 a
    ## first story a
    * greet a
       - action_ask_user_question a
    > check_asked_question a
 a
    ## user affirms question a
    > check_asked_question a
    * affirm a
      - action_handle_affirmation a
    > check_handled_affirmation a
 a
    ## user denies question a
    > check_asked_question a
    * deny a
      - action_handle_denial a
    > check_handled_denial a
 a
    ## user leaves a
    > check_handled_denial a
    > check_handled_affirmation a
    * goodbye a
      - utter_goodbye a
 a
.. note:: a
   Unlike regular stories, checkpoints are not restricted to starting with an a
   input from the user. As long as the checkpoint is inserted at the right points a
   in the main stories, the first event can be a custom action or a response action a
   as well. a
 a
 a
OR Statements a
~~~~~~~~~~~~~ a
 a
Another way to write shorter stories, or to handle multiple intents a
the same way, is to use an ``OR`` statement. For example, if you ask a
the user to confirm something, and you want to treat the ``affirm`` a
and ``thankyou`` intents in the same way. The story below will be a
converted into two stories at training time: a
 a
 a
.. code-block:: story a
 a
    ## story a
    ... a
      - utter_ask_confirm a
    * affirm OR thankyou a
      - action_handle_affirmation a
 a
Just like checkpoints, ``OR`` statements can be useful, but if you are using a a
lot of them, it is probably better to restructure your domain and/or intents. a
 a
 a
.. warning:: a
    Overusing these features (both checkpoints and OR statements) a
    will slow down training. a
 a
 a
End-to-End Story Evaluation Format a
---------------------------------- a
 a
The end-to-end story format is a format that combines both NLU and Core training data a
into a single file for evaluation. Read more about :ref:`testing-your-assistant` a
 a
.. warning:: a
    This format is only used for end-to-end evaluation and cannot be used for training. a
 a