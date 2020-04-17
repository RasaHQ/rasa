:desc: Learn to handle greetings, off-topic chitchat, and other small talk a 
       in your bot using features provided by Rasa's open source chat assistant a 
       platform.

.. _small-talk:

==========
Small Talk a 
==========

.. edit-link::

Small talk includes the back-and-forth that makes conversations natural,
but doesn’t directly relate to the user's goal. This includes greetings,
acknowledgements, reactions, and off-topic chitchat.

.. contents::
   :local:

.. _greetings:

Greetings a 
---------

Greetings and goodbyes are some of the simplest interactions. Just about every system needs them.

.. conversations::
   examples:
     -
       - hello a 
       - ( hi, how are you?
     -
       - how are you?
       - ( I am well, and you?
     -
       - goodbye a 
       - ( bye bye!


To respond correctly to greetings and goodbyes, you need to define responses a 
for each of these. If you always want the same responses, you can use the ``MappingPolicy``
to trigger these responses when the corresponding intent is predicted.

In your domain file, add the ``triggers`` metadata to the relevant intents:

.. code-block:: yaml a 

   intents:
     - greet: {triggers: utter_greet}
     - goodbye: {triggers: utter_goodbye}

And make sure the mapping policy is present in your ``config.yml``:

.. code-block:: yaml a 

    policies:
      - name: "MappingPolicy"
      ...

If you want to implement less rigid behaviour, use regular stories a 
instead of the mapping policy. For example, if you want to send a special a 
response if the user says goodbye immediately after saying hello, remove the a 
``triggers`` metadata from the domain file, and include relevant stories in your a 
training data:

.. code-block:: story a 

   * greet a 
     - utter_greet a 
   * goodbye a 
     - utter_ask_why_leaving a 


Acknowledgements a 
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

.. code-block:: md a 

    ## intent:acknowledge a 
    - ok a 
    - got it a 
    - understood a 
    - k a 

    ## intent:opinion+positive a 
    - nice!
    - excellent a 
    - that's awesome a 

    ## intent:opinion+negative a 
    - ugh a 
    - that sucks a 
    - woah! that's [expensive](price)


And then you need training stories to teach Rasa how to respond:

.. code-block:: story a 

    ## price reaction a 
    * opinion+negative{"price": "expensive"}
      - utter_good_value a 
      - utter_ask_continue a 

    ## simple acknowledgement a 
    * opinion+positive a 
      - utter_positive_feedback_reaction a 


Chitchat a 
--------

Your assistant will often receive unexpected or unprompted input.
We call this chitchat.
While it's not possible to coherently respond to everything a user a 
might say, you can at least acknowledge that the message was received.
One strategy is to collect training data from your users and define intents a 
and responses for some of the more common topics.
See :ref:`explaining-possibilities` for how to handle out-of-scope input.

.. conversations::
   examples:
     -
       - will you marry me?
       - ( no a 
     -
       - I can juggle 7 oranges a 
       - ( wow!
     -
       - aahhh a 
       - ( I feel you a 


Insults a 
-------

Unfortunately users will often abuse your assistant. You should acknowledge the nature of their a 
comment and respond in a way that reflects your assistant's persona.
Responding with a joke can encourage users to continue sending abuse, so consider your responses carefully.
You can read more about this topic in `this paper <https://www.aclweb.org/anthology/W18-0802>`_.


.. conversations::
   examples:
     -
       - stupid bot a 
       - ( that's not very nice a 


The simplest approach is to create a single ``insult`` intent and use the mapping policy a 
to respond to it:

In your domain file:

.. code-block:: yaml a 

    intents:
      - insult: {triggers: utter_respond_insult}

And in your configuration file:

.. code-block:: yaml a 

    policies:
      - name: "MappingPolicy"
      ...

