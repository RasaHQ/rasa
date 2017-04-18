.. _section_dataformat:

Training Data Format
====================

In the rasa NLU data format, there are three lists of examples: ``common_examples``, ``intent_examples``, and ``entity_examples``.
The ``common_examples`` are used to train both the entity and the intent models. 
In many cases it's fine to put all of your training examples in there. 
However, if you need lots and lots of examples to train a good entity recogniser, that can mess up 
your intent model because your classes are totally unbalanced. In that case it makes sense
to split up these lists. 

Entities are specified with a ``start`` and  ``end`` value, which together make a python style range to apply to the string, e.g. in the example below, with ``text="show me chinese restaurants"``, then ``text[8:15] == 'chinese'``.
Entities can span multiple words, and in fact the ``value`` field doesn't have to correspond exactly to the substring in your example. That way you can map syonyms, or misspellings, to the same ``value``.


.. code-block:: json

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

Entity Synonyms
---------------
If you define entities as having the same value they will be treated as synonyms. Here is an example of that:

.. code-block:: json

    [
      {
        "text": "in the center of NYC",
        "intent": "search",
        "entities": [
          {
            "start": 17,
            "end": 20,
            "value": "New York City",
            "entity": "city"
          }
        ]
      },
      {
        "text": "in the centre of New York City",
        "intent": "search",
        "entities": [
          {
            "start": 17,
            "end": 30,
            "value": "New York City",
            "entity": "city"
          }
        ]
      }
    ]

as you can see, the entity ``city`` has the value ``New York City`` in both examples, even though the text in the first
example states ``NYC``. By defining the value attribute to be different from the value found in the text between start
and end index of the entity, you can define a synonym. Whenever the same text will be found, the value will use the
synonym instead of the actual text in the message.

To use the synonyms defined in your training data, you need to make sure the pipeline contains the ``ner_synonyms``
component (see :ref:`section_pipeline`).