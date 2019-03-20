:desc: Check the architecture to understand how Rasa Core uses machine
       learning, context and state of the conversation to predict the
       next action of the AI Assistant.

.. _architecture:

High-Level Architecture
=======================


This diagram shows the basic steps of how a Rasa Core app
responds to a message:

.. image:: _static/images/rasa_arch_colour.png

The steps are:

1. The message is received and passed to an ``Interpreter``, which
   converts it into a dictionary including the original text, the intent,
   and any entities that were found.
2. The ``Tracker`` is the object which keeps track of conversation state.
   It receives the info that a new message has come in.
3. The policy receives the current state of the tracker.
4. The policy chooses which action to take next.
5. The chosen action is logged by the tracker.
6. A response is sent to the user.


.. note::

  Messages can be text typed by a human, or structured input
  like a button press.


The process is handled by the :class:`rasa.core.agent.Agent` class.


.. include:: feedback.inc
