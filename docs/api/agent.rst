Agent
=====

The agent allows you to train a model, load, and use it. It is a facade to
access most of Rasa Core's functionality using a simple API.

.. note::

    Not all functionality is exposed through methods on the
    agent. Sometimes you need to orchestrate the different components (domain,
    policy, interpreter, and the tracker store) on your own to customize them.

Here we go:

.. automodule:: rasa_core.agent
    :members:
    :undoc-members:
    :show-inheritance:
