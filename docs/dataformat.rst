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
