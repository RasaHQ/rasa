:desc: Read how to use the agent api to control most functionalities like
       training, loading and using a model of open source chatbot framework
       Rasa Stack.

.. _agent:

Agent
=====

The agent allows you to train a model, load, and use it. It is a simple API
that lets you access most of Rasa Core's functionality.

.. note::

    Not all functionality is exposed through methods on the
    agent. Sometimes you need to orchestrate the different components (domain,
    policy, interpreter, and the tracker store) on your own to customize them.


.. autoclass:: rasa.core.agent.Agent
    :members:


.. include:: ../feedback.inc
