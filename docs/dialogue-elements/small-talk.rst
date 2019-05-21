.. _small-talk:

==========
Small Talk
==========

Small talk includes the back-and-forth that makes conversations natural,
but doesn’t directly relate to the user's goal. This includes greetings,
acknowledgements, reactions, and off-topic chitchat.

.. contents::
   :local:

.. _greetings:

Greetings
---------

Greetings and goodbyes are some of the simplest interactions. Just about every system needs them.

.. conversations::
   examples:
     -
       - hello
       - ( hi, how are you?
     -
       - how are you?
       - ( I am well, and you?
     -
       - goodbye
       - ( bye bye!


To respond correctly to greetings and goodbyes, you need to define responses
for each of these. If you always want the same responses, you can use the ``MappingPolicy``
to trigger these responses when the corresponding intent is predicted.

In your domain file, add the ``triggers`` metadata to the relevant intents:

.. code-block:: yaml

   intents:
     - greet: {triggers: utter_greet}
     - goodbye: {triggers: utter_goodbye}

And make sure the mapping policy is present in your ``config.yml``:

.. code-block:: yaml

    policies:
      - name: "MappingPolicy"
      ...

If you want to implement less rigid behaviour, use regular stories
instead of the mapping policy. For example, if you want to send a special
response if the user says goodbye immediately after saying hello, remove the
``triggers`` metadata from the domain file, and include relevant stories in your
training data:

.. code-block:: story

   * greet
     - utter_greet
   * goodbye
     - utter_ask_why_leaving


Acknowledgements
----------------

Your users will often react to the things your assistant says, and will expect an acknowledgement.
Acknowledgements can be as simple as a thumbs up.
They reassure the user that their message has been received.
For the most common reactions, it is worth implementing specific responses.

.. conversations::
   examples:
     -
       - woah that's expensive!
       - ( we offer good value.
       - ( would you like to continue getting a quote?
     -
       - that's awesome!
       - ( glad you think so :)


First, you need NLU data for reactions and acknowledgements:

.. code-block:: md

    ## intent:acknowledge
    - ok
    - got it
    - understood
    - k

    ## intent:opinion+positive
    - nice!
    - excellent
    - that's awesome

    ## intent:opinion+negative
    - ugh
    - that sucks
    - woah! that's [expensive](price)


And then you need training stories to teach Rasa how to respond:

.. code-block:: story

    ## price reaction
    * opinion+negative{"price": "expensive"}
      - utter_good_value
      - utter_ask_continue

    ## simple acknowledgement
    * opinion+positive
      - utter_positive_feedback_reaction


Chitchat
--------

Your assistant will often receive unexpected or unprompted input.
We call this chitchat.
While it's not possible to coherently respond to everything a user
might say, you can at least acknowledge that the message was received.
One strategy is to collect training data from your users and define intents
and responses for some of the more common topics.
See :ref:`explaining-possibilities` for how to handle out-of-scope input.

.. conversations::
   examples:
     -
       - will you marry me?
       - ( no
     -
       - I can juggle 7 oranges
       - ( wow!
     -
       - aahhh
       - ( I feel you


Insults
-------

Unfortunately users will often abuse your assistant. You should acknowledge the nature of their
comment and respond in a way that reflects your assistant's persona.
Responding with a joke can encourage users to continue sending abuse, so consider your responses carefully.
You can read more about this topic in `this paper <https://www.aclweb.org/anthology/W18-0802>`_.


.. conversations::
   examples:
     -
       - stupid bot
       - ( that's not very nice


The simplest approach is to create a single ``insult`` intent and use the mapping policy
to respond to it:

In your domain file:

.. code-block:: yaml

    intents:
      - insult: {triggers: utter_respond_insult}

And in your configuration file:

.. code-block:: yaml

    policies:
      - name: "MappingPolicy"
      ...
