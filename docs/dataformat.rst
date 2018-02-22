.. _section_dataformat:

Training Data Format
====================

The training data for rasa NLU is structured into different parts, ``common_examples``, ``entity_synonyms`` and ``regex_features``. The most important one is ``common_examples``.

.. code-block:: json

    {
        "rasa_nlu_data": {
            "common_examples": [],
            "regex_features" : [],
            "entity_synonyms": []
        }
    }

The ``common_examples`` are used to train both the entity and the intent models. You should put all of your training
examples in the ``common_examples`` array. The next section describes in detail how an example looks like.
Regex features are a tool to help the classifier detect entities or intents and improve the performance.

You can use `Chatito <https://rodrigopivi.github.io/Chatito/>`__ , a tool for generating training datasets in rasa's format using a simple DSL.

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
That way you can map synonyms, or misspellings, to the same ``value``.

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

.. note::
    Please note that adding synonyms using the above format does not improve the model's classification of those entities.
    **Entities must be properly classified before they can be replaced with the synonym value.**


Regular Expression Features
---------------------------
Regular expressions can be used to support the intent classification and entity extraction. E.g. if your entity
has a certain structure as in a zipcode, you can use a regular expression to ease detection of that entity. For
the zipcode example it might look like this:

.. code-block:: json

    {
        "rasa_nlu_data": {
            "regex_features": [
                {
                    "name": "zipcode",
                    "pattern": "[0-9]{5}"
                },
                {
                    "name": "greet",
                    "pattern": "hey[^\\s]*"
                },
            ]
        }
    }

The name doesn't define the entity nor the intent, it is just a human readable description for you to remember what
this regex is used for. As you can see in the above example, you can also use the regex features to improve the intent
classification performance.

Try to create your regular expressions in a way that they match as few words as possible. E.g. using ``hey[^\s]*``
instead of ``hey.*``, as the later one might match the whole message whereas the first one only matches a single word.

Regex features for entity extraction are currently only supported by the ``ner_crf`` component! Hence, other entity
extractors, like ``ner_mitie`` won't use the generated features and their presence will not improve entity recognition
for these extractors. Currently, all intent classifiers make use of available regex features.

.. note::
    Regex features don't define entities nor intents! They simply provide patterns to help the classifier
    recognize entities and related intents. Hence, you still need to provide intent & entity examples as part of your
    training data!

Markdown Format
---------------------------

Alternatively training data can be used in the following markdown format. Examples are listed using the unordered
list syntax, e.g. minus ``-``, asterisk ``*``, or plus ``+``:

.. code-block:: markdown

    ## intent:check_balance
    - what is my balance <!-- no entity -->
    - how much do I have on my [savings](source_account) <!-- entity "source_account" has value "savings" -->
    - how much do I have on my [my savings account](source_account:savings) <!-- synonyms, method 1-->

    ## intent:greet
    - hey
    - hello

    ## synonym:savings   <!-- synonyms, method 2 -->
    - pink pig


    ## regex:zipcode
    - [0-9]{5}

Organization
---------------------------

The training data can either be stored in a single file or split into multiple files.
For larger training examples, splitting the training data into multiple files, e.g. one per intent, increases maintainability.

Storing files with different file formats, i.e. mixing markdown and JSON, is currently not supported.

.. note::
    Splitting the training data into multiple files currently only works for markdown and JSON data.
    For other file formats you have to use the single-file approach.

