:desc: Check the architecture to understand how Rasa Core uses machine
       learning, context and state of the conversation to predict the
       next action of the AI Assistant.

.. _architecture:

Architecture
============


Message Handling
^^^^^^^^^^^^^^^^

This diagram shows the basic steps of how an assistant built with Rasa Stack
responds to a message:

.. image:: ../_static/images/rasa-message-processing.png

The steps are:

1. The message is received and passed to an ``Interpreter``, which
   converts it into a dictionary including the original text, the intent,
   and any entities that were found. This part is handled by NLU.
2. The ``Tracker`` is the object which keeps track of conversation state.
   It receives the info that a new message has come in.
3. The policy receives the current state of the tracker.
4. The policy chooses which action to take next.
5. The chosen action is logged by the tracker.
6. A response is sent to the user.


.. note::

  Messages can be text typed by a human, or structured input
  like a button press.
