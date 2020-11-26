:desc: Dialogue elements are an abstraction layer for your conversational AI platform 
       which describe common, recurring patterns in chatbot conversations.

.. _dialogue-elements:

Dialogue Elements
=================

.. edit-link::

Dialogue elements are common conversation patterns.
We use three different levels of abstraction to discuss AI assistants.
This can be helpful in a product team, so that you have a common language
which designers, developers, and product owners can use to discuss 
issues and new features.

- highest level: user goals
- middle level: dialogue elements
- lowest level: intents, entities, actions, slots, and templates.



.. note::
   Some chatbot tools use the word ``intent`` to refer to the user
   goal. This is confusing because only some messages tell you what a user's
   goal is. If a user says "I want to open an account" (``intent: open_account``),
   that is clearly their goal. But most user messages ("yes", "what does that mean?", "I don't know")
   aren't specific to one goal. In Rasa, every message has an intent,
   and a user goal describes what a person wants to achieve.
   

.. image:: /_static/images/intents-user-goals-dialogue-elements.png


