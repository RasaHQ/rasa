.. _message_handling:

Message Handling
================

In most cases, your bot receives messages over a network.
This means you have an outward facing web server which accepts requests.
Responding to these requests synchronously isn't a very scalable solution.

The way Rasa Core handles messages is slightly different.
The ``Controller`` appends messages to a queue, and creates a ``MessageHandler``
which works through the messages in the queue.

This makes it possible to safely scale to multiple Rasa DM instances.
By default Rasa uses the ``InMemoryMessageQueue``, which can't be shared among processes,
but this can easily be swapped to an external message broker.

.. image:: _static/images/message_handling.png
