:desc: Dialogue elements are an abstraction layer for your conversational AI platform  a
       which describe common, recurring patterns in chatbot conversations. a
 a
.. _dialogue-elements: a
 a
Dialogue Elements a
================= a
 a
.. edit-link:: a
 a
Dialogue elements are common conversation patterns. a
We use three different levels of abstraction to discuss AI assistants. a
This can be helpful in a product team, so that you have a common language a
which designers, developers, and product owners can use to discuss  a
issues and new features. a
 a
- highest level: user goals a
- middle level: dialogue elements a
- lowest level: intents, entities, actions, slots, and responses. a
 a
 a
 a
.. note:: a
   Some chatbot tools use the word ``intent`` to refer to the user a
   goal. This is confusing because only some messages tell you what a user's a
   goal is. If a user says "I want to open an account" (``intent: open_account``), a
   that is clearly their goal. But most user messages ("yes", "what does that mean?", "I don't know") a
   aren't specific to one goal. In Rasa, every message has an intent, a
   and a user goal describes what a person wants to achieve. a
    a
 a
.. image:: /_static/images/intents-user-goals-dialogue-elements.png a
 a
 a
 a