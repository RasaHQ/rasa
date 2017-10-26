.. _interpreters:

Interpreters
============

The job of interpreting text is mostly outside the scope of Rasa Core.
To turn text into structured data you can use Rasa NLU, or a cloud service like wit.ai.
If your bot uses button clicks or other input which isn't natural language, you don't need
an interepreter at all. You can define your own ``Interpreter`` subclass which does any custom
logic you may need. You can look at the ``RegexInterpreter`` class as an example.


To use something other than Rasa NLU, you just need to implement a subclass of ``Interpreter``
which has a method ``parse(message)`` which takes a single string argument and returns a dict in the following format:


.. code-block:: javascript

    {
      "text": "show me chinese restaurants",
      "intent": "restaurant_search",
      "entities": [
        {
          "start": 8,
          "end": 15,
          "value": "chinese",
          "entity": "cuisine"
        }
      ]
    }

.. note: the ``"start"`` and ``"start"`` values in the entities are optional


Regex
------

For testing and for writing stories, Rasa Core has a ``RegexInterpreter``.
This matches strings in the format ``_intent[entity1=value, entity2=value]``.