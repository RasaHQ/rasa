.. _plumbing:

Plumbing - How it all fits together
===================================


This diagram shows the basic steps of how a Rasa Core app responds to a message:

.. image:: _static/images/rasa_arch_colour.png

The steps are: 

1. The message is received and passed to an ``Interpreter``, which converts it into a dictionary
   including the original text, the intent, and any entities that were found.
2. The ``Tracker`` is the object which keeps track of conversation state. 
   It receives the info that a new message has come in.
3. The policy receives the current state of the tracker
4. The policy chooses which action to take next.
5. The chosen action is logged by the tracker
6. A response is sent to the user


.. note:: Messages can be text typed by a human, but can equally well be a button payload, or a message that's already been interpreted (as you would get from an Amazon Echo). See the section on :ref:`interpreters` for details.

The steps above are carried out by the ``MessageProcessor``, which receives
the messages and processes them as they arrive.

Feel free to check out the :doc:`tutorial_basics` and
:doc:`tutorial_interactive_learning` to see a bot in action!