.. _section_dataformat:

Training Data Format
====================

The training data for rasa NLU is structured into different parts. The most important one is ``common_examples``.

.. code-block:: json

    {
        "rasa_nlu_data": {
            "common_examples": []
        }
    }

The ``common_examples`` are used to train both the entity and the intent models. You should put all of your training
examples in the ``common_examples`` array. The next section describes in detail how an example looks like.

Common Examples
---------------

Common examples have three components: ``text``, ``intent``, and ``entities``. The first two are strings while the last one is an array.

 - The *text* is the search query; An example of what would be submitted for parsing. [required]
 - The *intent* is the intent that should be associated with the text. [optional]
 - The *entities* are specific parts of the text which need to be identified. [optional]

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

Alternatively, you can add an "entity_synonyms" array to define several synonyms to one entity value. Here is an example of that:

.. code-block:: json

  {
    "rasa_nlu_data": {
      "entity_synonyms": [
        {
          "value": "New York City",
          "synonyms": ["NYC", "nyc", "the big apple"]
        }
      ]
    }
  }
