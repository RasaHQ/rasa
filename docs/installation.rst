.. _installation:

Installation and Hello World
============================

Installation
------------

To install, run:

.. code-block:: bash
   
   git clone https://github.com/rasahq/rasa_core.git
   cd rasa_core
   pip install -r requirements.txt
   pip install -e .


Additional Dependencies
-----------------------

We use Rasa NLU for intent classification & entity extraction, but you are free to use other NLU services like wit.ai, api.ai, or LUIS.ai. In fact, you don't need to use NLU at all, if your messaging app uses buttons rather than free text.

In addition, you will probably want to use a machine learning library to power one of the policies.
Hopefully the requirements for using ``sklearn_policy.py`` and ``keras_policy.py`` are self-evident.

Hello, World!
-------------

First things first, let's try it out! From the project's root dir, run:

.. code-block:: bash

    python examples/hello_world/run.py
    Bot loaded. Type hello and press enter : 
    hello
    hey there!

