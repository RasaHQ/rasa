:desc: Leverage information from knowledge bases inside conversations using KnowledgeBaseActions
       in open source bot framework Rasa.

.. _knowledge_bases:

Knowledge Base Actions
======================

.. edit-link::

.. warning::
   This feature is experimental.

.. contents::
   :local:


TODO:
- just one knowledge base at a time can be used
- limitations of implementation


A lot of users want to obtain detailed information about certain objects, such as restaurants or hotels, during a conversation.
Users, for example, want to know restaurants in Berlin.
They want to obtain details, such as, if an restaurant has outside seating or how expensive the restaurant is.
In order to answer those user requests domain knowledge is needed.
Hard-coding the information would not help as the information are subject to change.
Additionally, users do not only refer to objects by their names, but also use terms, such as "the first one" or "that
restaurant", to refer to a specific object.
Objects need to be recognised and reused at a later point in the conversation.

To handle the above challenges, we recommend that you create a custom action that inherits from ``ActionQueryKnowledgeBase``.
This is a single actions which contains the logic to query a knowledge base for objects and their attributes.
The action is also able to resolve certain mention of objects, such as ``the first one`` or ``that restaurant``.
You can find a complete example in ``examples/knowledge_base_bot``.

.. note::
   There is a tutorial `here <https://blog.rasa.com/integrating-rasa-with-knowledge-bases/>`_ about how to use
   knowledge bases in custom actions. The tutorial will explain in detail the implementation behind
   ``ActionQueryKnowledgeBase``.


Create a KnowledgeBase
----------------------

The data you will use to answer the user's request comes from a knowledge base.
A knowledge base can be used to store complex data structures.
If you just have some data points that fit into memory, you can use our ``InMemoryKnowledgeBase`` implementation.
To initialize an ``InMemoryKnowledgeBase`` you need to provide the data into a json file.

Let's take a look at an example:

.. code-block:: json

    {
        "restaurant": [
            {
                "id": 0,
                "name": "Donath",
                "cuisine": "Italian",
                "outside-seating": true,
                "price-range": "mid-range"
            },
            {
                "id": 1,
                "name": "Berlin Burrito Company",
                "cuisine": "Mexican",
                "outside-seating": false,
                "price-range": "cheap"
            },
            {
                "id": 2,
                "name": "I due forni",
                "cuisine": "Italian",
                "outside-seating": true,
                "price-range": "mid-range"
            }
        ],
        "hotel": [
            {
                "id": 0,
                "name": "Hilton",
                "price-range": "expensive",
                "breakfast-included": true,
                "city": "Berlin",
                "free-wifi": true,
                "star-rating": 5,
                "swimming-pool": true
            },
            {
                "id": 1,
                "name": "Hilton",
                "price-range": "expensive",
                "breakfast-included": true,
                "city": "Frankfurt am Main",
                "free-wifi": true,
                "star-rating": 4,
                "swimming-pool": false
            },
            {
                "id": 2,
                "name": "B&B",
                "price-range": "mid-range",
                "breakfast-included": false,
                "city": "Berlin",
                "free-wifi": false,
                "star-rating": 1,
                "swimming-pool": false
            },
        ]
    }

The above json file contains data about restaurants and hotels.
The json structure should contain a key for every object type, i.e. "restaurant" and "hotel" in the example.
Every object type maps to a list of objects.

Once the data are defined in a json file, called, for example, ``data.json``, you can create your
``InMemoryKnowledgeBase``:

.. code-block:: python

    knowledge_base = InMemoryKnowledgeBase.load("data.json")

Customize your InMemoryKnowledgeBase
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The class ``InMemoryKnowledgeBase`` inherits ``KnowledgeBase``.
You can customize your ``InMemoryKnowledgeBase`` by overwriting the following functions:

- ``get_key_attribute_of_object``: To keep track of what object was talking about last, we store the value of the
  key attribute of that object in a specific slot. Every object should have a key attribute that is unique, i.e.
  similar to the primary key in a relation database. Per default the name of the key attribute for every object type
  is set to "id". You can overwrite the name of the key attribute for a specific object type by calling
  ``set_key_attribute_of_object()``.
- ``get_representation_function_of_object``: This methods returns a lambda function that maps an object from the
  knowledge base to a string representation. This function is used whenever an object is outputted to the user.
  Per default the lambda function is set to ``lambda obj: obj["name"]``. So, it returns the value of the attribute
  "name" of the object. If your object does not have an attribute "name", or the "name" of an object might not be
  unambiguous, you should set a new lambda function for that object type by calling
  ``set_representation_function_of_object()``.
- ``set_ordinal_mention_mapping``: The ordinal mention mapping is needed to resolve an ordinal mention to an object
  in a list. For example, if the bot listed a few restaurants in Berlin, and the user then asked "Does the second one
  have outside seating?", you need to resolve "second one" to the corresponding object the bot listed before. Per
  default the ordinal mention mapping looks like this:

  .. code-block:: python

      {
          "1": lambda l: l[0],
          "2": lambda l: l[1],
          "3": lambda l: l[2],
          "4": lambda l: l[3],
          "5": lambda l: l[4],
          "6": lambda l: l[5],
          "7": lambda l: l[6],
          "8": lambda l: l[7],
          "9": lambda l: l[8],
          "10": lambda l: l[9],
          "ANY": lambda l: random.choice(list),
          "LAST": lambda l: l[-1],
      }

  You can overwrite it by calling the function ``set_ordinal_mention_mapping``.
  More on the usage can be found in section :ref:`resolve_mentions`.

Creating your own Knowledge Base Implementation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you have more data or if you want to use a more complex data structure that, for example, involves relations between
different objects, you can also create your own knowledge base implementation.
Just inherit ``KnowledgeBase`` and implement the methods ``get_objects()``, ``get_object()``, and
``get_attributes_of_object()``.
You can also customize your knowledge base further, for example, by adapting the methods mentioned in the previous
section.

.. note::
   We wrote a `blog post <https://blog.rasa.com/set-up-a-knowledge-base-to-encode-domain-knowledge-for-rasa/>`_
   that explains how you can set up your own knowledge base.

Create an ActionQueryKnowledgeBase
----------------------------------

Whenever you create an ``ActionQueryKnowledgeBase``, you need to pass a ``KnowledgeBase`` to the constructor.
It can be either an ``InMemoryKnowledgeBase`` or your own implementation of a ``KnowledgeBase``.
To create your own knowledge base action, you need to inherit ``ActionQueryKnowledgeBase`` and pass the knowledge
base to the constructor of ``ActionQueryKnowledgeBase``.

.. code-block:: python

    class MyKnowledgeBaseAction(ActionQueryKnowledgeBase):
        def __init__(self):
            knowledge_base = InMemoryKnowledgeBase(schema, data)
            super().__init__(knowledge_base)

You don't need to do anything else.
The action is already able to query the knowledge base.
The name of the action is ``action_query_knowledge_base``.
Don't forget to add it to your domain file.

.. note::
   If you overwrite the default action name ``action_query_knowledge_base``, you need to add the following three
   slots to your domain file: ``knowledge_base_objects``, ``knowledge_base_last_object``, and
   ``knowledge_base_last_object_type``.
   The slots are used internally by ``ActionQueryKnowledgeBase``.
   If you keep the default action name, those slots will be added automatically for you.


Defining the NLU Data
---------------------

To be able to understand that the user wants to retrieve some information from the knowledge base, you need to define
a new intent, for example, ``query_knowledge_base``.
The intent should contain all kind of user requests.

Let's look at an example:

.. code-block:: yaml

    ## intent:query_knowledge_base
    - what [restaurants](object_type:restaurant) can you recommend?
    - list some [restaurants](object_type:restaurant)
    - can you name some [restaurants](object_type:restaurant) please?
    - can you show me some [restaurant](object_type:restaurant) options
    - list [German](cuisine) [restaurants](object_type:restaurant)
    - do you have any [mexican](cuisine) [restaurants](object_type:restaurant)?
    - do you know the [price range](attribute:price-range) of [that one](mention)?
    - what [cuisine](attribute) is it?
    - do you know what [cuisine](attribute) the [last one](mention:LAST) has?
    - does [Donath](restaurant) have [outside seating](attribute:outside-seating)?
    - what is the [price range](attribute:price-range) of [Berlin Burrito Company](restaurant)?
    - what is with [I due forni](restaurant)?
     ...

The above examples just show examples related to the restaurant domain.
You should add examples for every object type that exists in your knowledge base.

All user requests can be divided into two categories:
(1) The user wants to obtain a list of objects of a specific type or (2) the user wants to know about a certain
attribute of an object.
The ``ActionQueryKnowledgeBase`` can handle both of those requests.
Other requests, such as comparison between objects, are currently not supported.

Another thing you may have noticed is, that we marked different kind of entities in the NLU data.
If you want to use ``ActionQueryKnowledgeBase``, you need to specify the following entities:

- ``object_type``: Whenever the user is talking about a specific object type from your knowledge base, the type should
  be extracted by the NER. Use :ref:`entity_synonyms` to map, for example, "restaurants" to the correct object type
  listed in the knowledge base, e.g. "restaurant".
- ``mention``: If the user refers to an object via "the first one", "that one", or "it", you should mark those terms
  as ``mention``. We also use :ref:`entity_synonyms` to map some of the mentions to symbols. More on that in
  :ref:`resolve_mentions`.
- ``attribute``: All attribute names defined in your knowledge base should be marked in the NLU data. Again, use
  :ref:`entity_synonyms` to map variations of an attribute name to the one used in the knowledge base.

Don't forget to add those entities to your domain file (as entities and slots).

Query the Knowledge Base for Objects
------------------------------------

In order to query the knowledge base for any kind of objects, the user's request needs to include the object type.
Otherwise, the action does not know what objects the user is interested in and cannot formulate the query.

The user may restrict his request to a specific kind of object.
For example, he could say ``What Italian restaurant options in Berlin do I have?``.
In this example the user want to obtain a list of restaurants that (1) have an Italian cuisine and (2) are located in
Berlin.
In order to filter the objects in the knowledge base, you need to mark "Italian" and "Berlin" as entities.
E.g. ``What [Italian](cuisine) [restaurant](object_type) options in [Berlin](city) do I have?``.
The names of the attributes, e.g. "cuisine" and "city", should be equal to the ones used in the knowledge base.
You also need to add those entities as entities and slots in the domain file.
If the NER detects those attributes in the request of the user, the action will use those for filter the restaurants.

Once the bot retrieved some entities from the knowledge base, it will response to the user with

    `Found the following objects of type 'restaurant':`
    `1: I due forni`
    `2: PastaBar`
    `3: Berlin Burrito Company`

Or if no entities could be found

    `I could not find any objects of type 'restaurant'.`

If you want to change the utterance of the bot, you can overwrite the method ``utter_objects()`` in your action.

Query the Knowledge Base for an Attribute of an Object
------------------------------------------------------

To obtain the value of an attribute for a specific object from the knowledge base, the action needs to know the object
and attribute of interest.
Every object has a key attribute which should be unique.
Thus, we use the value of that key attribute to identify an object.
The user can either refer to the object of interest by its name, e.g. value of the key attribute, or he refers to a
previously mentioned object.
See the next section on how we resolve mentions to the actual object.
The attribute of interest should be included in the user's request.
For example, ``What is the cuisine of PastaBar?``, contains the attribute of interest "cuisine" and the object of
interest "PastaBar".
Both should be marked as entities in the NLU training data, e.g. ``What is the [cuisine](attribute) of [PastaBar](restaurant)?``.

If the attribute was found in the knowledge base, the bot will response with the following utterance:

    `'PastaBar' has the value 'Italian' for attribute 'cuisine'.`

If no value for the requested attribute was found, the bot will response with

    `Did not found a valid value for attribute 'cuisine' for object 'PastaBar'.`

If you want to change the utterance of the bot, you can overwrite the method ``utter_attribute_value()``.

.. _resolve_mentions:

Resolve Mentions
----------------

The user may refer to previously mentioned objects during the conversation.
Users can refer to objects in many different ways.
Our action is able to (1) resolve ordinal mentions, such as "the first one", to the actual object and (2) resolve any
other mention, such as "it" or "that one" to the last mentioned object in the conversation.

Ordinal Mentions
~~~~~~~~~~~~~~~~
If the user refers to an object by its position in a list, we talk about ordinal mentions.
Examples for ordinal mentions are

- the first one
- the last one
- any
- 4

Ordinal mentions are typically used when a list of objects was presented to the user.
To resolve those mentions to the actual object, we use an ordinal mention mapping which is set in the ``KnowledgeBase``
class.
The ordinal mention mapping maps a string, such as "1", to the object in a list, e.g. ``lambda l: l[0]``.
You can overwrite the ordinal mention mapping by calling the function ``set_ordinal_mention_mapping()`` on your
``KnowledgeBase`` implementation.

Other Mentions
~~~~~~~~~~~~~~
Take a look at the following conversation:

- User: What is the cuisine of PastaBar?
- Bot: PastaBar has an Italian cuisine.
- User: Does it have wifi?
- Bot: Yes.
- User: Can you give me an address?

In the second utterance of the user, the user refers to "PastaBar" by the word "it".
If the NER detected "it" as the entity ``mention``, the knowledge base action would resolve it to the last mentioned
object in the conversation, e.g. "PastaBar".
In the next utterance of the user, the user refers indirect to the object "PastaBar".
However, the user does not mention "PastaBar" explicit.
The knowledge base action would detect that the user wants to obtain the value of a specific attribute.
If no mention or object could be detected by the NER, the action just assumes the user is talking about he last
mentioned object, e.g. "PastaBar".
You can disable this behaviour by setting ``use_last_object_mention`` to ``False`` when initializing the action.
