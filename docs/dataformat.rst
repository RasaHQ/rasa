.. _section_dataformat:

Training Data Format
====================

The training data for rasa NLU has three arrays inside of a top level object ``common_examples``, ``intent_examples``, and ``entity_examples``. Not all three are required, you can use each of them as needed by the model you are trying to train.

.. code-block:: json

    {
        "rasa_nlu_data": {
            "common_examples": [],
            "intent_examples": [],
            "entity_examples": []
        }
    }

The ``common_examples`` are used to train both the entity and the intent models while the other arrays target intents and entities exclusively.

In many cases it's fine to put all of your training examples in the ``common_examples`` array. 
However, if you need lots and lots of examples to train a good entity recogniser, that can mess up 
your intent model because your classes would become unbalanced. In that case it makes sense
to split up these lists.

Common Examples
---------------

Common examples have three components: ``text``, ``intent``, and ``entities``. The first two are strings while the last one is an array.

 - The *text* is the search query; An example of what would be submitted for parsing.
 - The *intent* is the intent that should be associated with the text.
 - The *entities* are specific parts of the text which need to be identified.

Entities are specified with a ``start`` and  ``end`` value, which together make a python
style range to apply to the string, e.g. in the example below, with ``text="show me chinese
restaurants"``, then ``text[8:15] == 'chinese'``. Entities can span multiple words, and in
fact the ``value`` field does not have to correspond exactly to the substring in your example.
That way you can map syonyms, or misspellings, to the same ``value``.

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
    
Intent Examples
---------------
Omit the entire entities array:

.. code-block:: json

    {
      "text": "show me chinese restaurants", 
      "intent": "restaurant_search"
    }

Entity Examples
---------------
Simply omit the ``intent`` section:

.. code-block:: json

    {
      "text": "show me chinese restaurants", 
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
