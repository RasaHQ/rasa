:desc: Check the architecture to understand how Rasa uses machine a
       learning, context and state of the conversation to predict the a
       next action of the AI Assistant. a
 a
.. _architecture: a
 a
Architecture a
============ a
 a
.. edit-link:: a
 a
 a
Message Handling a
^^^^^^^^^^^^^^^^ a
 a
This diagram shows the basic steps of how an assistant built with Rasa a
responds to a message: a
 a
.. image:: ../_static/images/rasa-message-processing.png a
 a
The steps are: a
 a
1. The message is received and passed to an ``Interpreter``, which a
   converts it into a dictionary including the original text, the intent, a
   and any entities that were found. This part is handled by NLU. a
2. The ``Tracker`` is the object which keeps track of conversation state. a
   It receives the info that a new message has come in. a
3. The policy receives the current state of the tracker. a
4. The policy chooses which action to take next. a
5. The chosen action is logged by the tracker. a
6. A response is sent to the user. a
 a
 a
.. note:: a
 a
  Messages can be text typed by a human, or structured input a
  like a button press. a
 a