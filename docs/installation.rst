.. _installation:

Installation and Hello World
============================

Installation
------------
The recommended way to install Rasa Core is using pip:

.. code-block:: bash

    pip install rasa_core

If you want to use the bleeding edge version use github + setup.py:

.. code-block:: bash

    git clone https://github.com/RasaHQ/rasa_core.git
    cd rasa_core
    pip install -r requirements.txt
    python setup.py install

.. note::
    If you want to change the Rasa Core code and want to run the tests or
    build the documentation, you need to install the development dependencies:

    .. code-block:: bash

        pip install -r dev-requirements.txt


Additional Dependencies
-----------------------

We use Rasa NLU for intent classification & entity extraction,
but you are free to use other NLU services like wit.ai, api.ai, or LUIS.ai. If you
want to use Rasa NLU make sure to follow the installation instructions of the
`NLU docs <https://nlu.rasa.ai>`_ as well.
In fact, you don't need to use NLU at all, if your messaging app uses buttons
rather than free text.

Hello, World!
-------------

First things first, let's try it out! From the project's root dir, run:

.. code-block:: bash

    python examples/hello_world/run.py
    Bot loaded. Type hello and press enter : 
    hello
    hey there!
