:desc: Read how Rasa Stack interprets a message or non-textual input like
       buttons or other structured input for contextual conversation design.

.. _interpreters:

Interpreters
============

Rasa Core itself does not interpret text. You can use `Rasa NLU <https://rasa.com/docs/nlu/>`_ for this.


.. autoclass:: rasa.core.interpreter.RasaNLUHttpInterpreter

   .. automethod:: RasaNLUHttpInterpreter.parse

.. autoclass:: rasa.core.interpreter.RasaNLUInterpreter

   .. automethod:: RasaNLUInterpreter.parse


To use something other than Rasa NLU, you just need to implement a
subclass of ``Interpreter``
which has a method ``parse(message)`` which takes a single string argument
and returns a dict in the following format:


.. code-block:: javascript

    {
      "text": "show me chinese restaurants",
      "intent": {
        "name": "restaurant_search",
        "confidence": 1.0
      }
      "entities": [
        {
          "start": 8,
          "end": 15,
          "value": "chinese",
          "entity": "cuisine"
        }
      ]
    }

.. note:

    The ``"start"`` and ``"end"`` values in the entities are optional

.. _fixed_intent_format:

Buttons and other Structured Input
----------------------------------

If your bot uses button clicks or other input which isn't natural language, you don't need
an interpreter at all. You can define your own ``Interpreter`` subclass which does any custom
logic you may need. You can look at the ``RegexInterpreter`` class as an example.


Sometimes, you want to make sure a message is treated as being of a fixed
intent containing defined entities. To achieve that, you can specify the
message in a markup format instead of using the text of the message.

Instead of sending a message like ``Hello I am Rasa`` and hoping that gets
classified correctly, you can circumvent the NLU and directly send the
bot a message like ``/greet{"name": "Rasa"}``. Rasa Core will treat this
incoming message like a normal message with the intent ``greet`` and the entity
``name`` with value ``Rasa``.

If you want to specify an input string, that contains multiple entity values of
the same type, you can use:

.. code-block:: bash

    /add_to_shopping_list{"item": ["milk", "salt"]}

Which corresponds to a message ``"I want to add milk and salt to my list"``.

If you want to specify the intent confidence, you can use:

.. code-block:: bash

    /deny@0.825

Which feeds the intent ``greet`` with confidence ``0.825``.  Combining confidence
and entities:

.. code-block:: bash

    /add_to_shopping_list@0.825{"item": ["milk", "salt"]}


.. include:: ../feedback.inc
