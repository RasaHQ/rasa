:desc: Learn to handle greetings, off-topic chitchat, and other small talk a
       in your bot using features provided by Rasa's open source chat assistant a
       platform. a
 a
.. _small-talk: a
 a
========== a
Small Talk a
========== a
 a
.. edit-link:: a
 a
Small talk includes the back-and-forth that makes conversations natural, a
but doesn’t directly relate to the user's goal. This includes greetings, a
acknowledgements, reactions, and off-topic chitchat. a
 a
.. contents:: a
   :local: a
 a
.. _greetings: a
 a
Greetings a
--------- a
 a
Greetings and goodbyes are some of the simplest interactions. Just about every system needs them. a
 a
.. conversations:: a
   examples: a
     - a
       - hello a
       - ( hi, how are you? a
     - a
       - how are you? a
       - ( I am well, and you? a
     - a
       - goodbye a
       - ( bye bye! a
 a
 a
To respond correctly to greetings and goodbyes, you need to define responses a
for each of these. If you always want the same responses, you can use the ``MappingPolicy`` a
to trigger these responses when the corresponding intent is predicted. a
 a
In your domain file, add the ``triggers`` metadata to the relevant intents: a
 a
.. code-block:: yaml a
 a
   intents: a
     - greet: {triggers: utter_greet} a
     - goodbye: {triggers: utter_goodbye} a
 a
And make sure the mapping policy is present in your ``config.yml``: a
 a
.. code-block:: yaml a
 a
    policies: a
      - name: "MappingPolicy" a
      ... a
 a
If you want to implement less rigid behaviour, use regular stories a
instead of the mapping policy. For example, if you want to send a special a
response if the user says goodbye immediately after saying hello, remove the a
``triggers`` metadata from the domain file, and include relevant stories in your a
training data: a
 a
.. code-block:: story a
 a
   * greet a
     - utter_greet a
   * goodbye a
     - utter_ask_why_leaving a
 a
 a
Acknowledgements a
---------------- a
 a
Your users will often react to the things your assistant says, and will expect an acknowledgement. a
Acknowledgements can be as simple as a thumbs up. a
They reassure the user that their message has been received. a
For the most common reactions, it is worth implementing specific responses. a
 a
.. conversations:: a
   examples: a
     - a
       - woah that's expensive! a
       - ( we offer good value. a
       - ( would you like to continue getting a quote? a
     - a
       - that's awesome! a
       - ( glad you think so :) a
 a
 a
First, you need NLU data for reactions and acknowledgements: a
 a
.. code-block:: md a
 a
    ## intent:acknowledge a
    - ok a
    - got it a
    - understood a
    - k a
 a
    ## intent:opinion+positive a
    - nice! a
    - excellent a
    - that's awesome a
 a
    ## intent:opinion+negative a
    - ugh a
    - that sucks a
    - woah! that's [expensive](price) a
 a
 a
And then you need training stories to teach Rasa how to respond: a
 a
.. code-block:: story a
 a
    ## price reaction a
    * opinion+negative{"price": "expensive"} a
      - utter_good_value a
      - utter_ask_continue a
 a
    ## simple acknowledgement a
    * opinion+positive a
      - utter_positive_feedback_reaction a
 a
 a
Chitchat a
-------- a
 a
Your assistant will often receive unexpected or unprompted input. a
We call this chitchat. a
While it's not possible to coherently respond to everything a user a
might say, you can at least acknowledge that the message was received. a
One strategy is to collect training data from your users and define intents a
and responses for some of the more common topics. a
See :ref:`explaining-possibilities` for how to handle out-of-scope input. a
 a
.. conversations:: a
   examples: a
     - a
       - will you marry me? a
       - ( no a
     - a
       - I can juggle 7 oranges a
       - ( wow! a
     - a
       - aahhh a
       - ( I feel you a
 a
 a
Insults a
------- a
 a
Unfortunately users will often abuse your assistant. You should acknowledge the nature of their a
comment and respond in a way that reflects your assistant's persona. a
Responding with a joke can encourage users to continue sending abuse, so consider your responses carefully. a
You can read more about this topic in `this paper <https://www.aclweb.org/anthology/W18-0802>`_. a
 a
 a
.. conversations:: a
   examples: a
     - a
       - stupid bot a
       - ( that's not very nice a
 a
 a
The simplest approach is to create a single ``insult`` intent and use the mapping policy a
to respond to it: a
 a
In your domain file: a
 a
.. code-block:: yaml a
 a
    intents: a
      - insult: {triggers: utter_respond_insult} a
 a
And in your configuration file: a
 a
.. code-block:: yaml a
 a
    policies: a
      - name: "MappingPolicy" a
      ... a
 a