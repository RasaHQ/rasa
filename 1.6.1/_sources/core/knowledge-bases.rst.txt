:desc: Leverage information from knowledge bases inside conversations using ActionQueryKnowledgeBase
       in open source bot framework Rasa.

.. _knowledge_base_actions:

Knowledge Base Actions
======================

.. edit-link::

.. warning::
   This feature is experimental.
   We introduce experimental features to get feedback from our community, so we encourage you to try it out!
   However, the functionality might be changed or removed in the future.
   If you have feedback (positive or negative) please share it with us on the `forum <https://forum.rasa.com>`_.

.. contents::
   :local:

Knowledge base actions enable you to handle the following kind of conversations:

.. image:: ../_static/images/knowledge-base-example.png

A common problem in conversational AI is that users do not only refer to certain objects by their names,
but also use reference terms such as "the first one" or "it".
We need to keep track of the information that was presented to resolve these mentions to
the correct object.

In addition, users may want to obtain detailed information about objects during a conversation --
for example, whether a restaurant has outside seating, or how expensive it is.
In order to respond to those user requests, knowledge about the restaurant domain is needed.
Since the information is subject to change, hard-coding the information isn't the solution.


To handle the above challenges, Rasa can be integrated with knowledge bases. To use this integration, you can create a
custom action that inherits from ``ActionQueryKnowledgeBase``, a pre-written custom action that contains
the logic to query a knowledge base for objects and their attributes.

You can find a complete example in ``examples/knowledgebasebot``
(`knowledge base bot <https://github.com/RasaHQ/rasa/blob/master/examples/knowledgebasebot/>`_), as well as instructions
for implementing this custom action below.


Using ``ActionQueryKnowledgeBase``
----------------------------------

.. _create_knowledge_base:

Create a Knowledge Base
~~~~~~~~~~~~~~~~~~~~~~~

The data used to answer the user's requests will be stored in a knowledge base.
A knowledge base can be used to store complex data structures.
We suggest you get started by using the ``InMemoryKnowledgeBase``.
Once you want to start working with a large amount of data, you can switch to a custom knowledge base
(see :ref:`custom_knowledge_base`).

To initialize an ``InMemoryKnowledgeBase``, you need to provide the data in a json file.
The following example contains data about restaurants and hotels.
The json structure should contain a key for every object type, i.e. ``"restaurant"`` and ``"hotel"``.
Every object type maps to a list of objects -- here we have a list of 3 restaurants and a list of 3 hotels.

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


Once the data is defined in a json file, called, for example, ``data.json``, you will be able use the this data file to create your
``InMemoryKnowledgeBase``, which will be passed to the action that queries the knowledge base.

Every object in your knowledge base should have at least the ``"name"`` and ``"id"`` fields to use the default implementation.
If it doesn't, you'll have to :ref:`customize your InMemoryKnowledgeBase <customize_in_memory_knowledge_base>`.


Define the NLU Data
~~~~~~~~~~~~~~~~~~~

In this section:

- we will introduce a new intent, ``query_knowledge_base``
- we will to annotate ``mention`` entities so that our model detects indirect mentions of objects like "the
  first one"
- we will use :ref:`synonyms <entity_synonyms>` extensively

For the bot to understand that the user wants to retrieve information from the knowledge base, you need to define
a new intent. We will call it ``query_knowledge_base``.

We can split requests that ``ActionQueryKnowledgeBase`` can handle into two categories:
(1) the user wants to obtain a list of objects of a specific type, or (2) the user wants to know about a certain
attribute of an object. The intent should contain lots of variations of both of these requests:

.. code-block:: md

    ## intent:query_knowledge_base
    - what [restaurants](object_type:restaurant) can you recommend?
    - list some [restaurants](object_type:restaurant)
    - can you name some [restaurants](object_type:restaurant) please?
    - can you show me some [restaurant](object_type:restaurant) options
    - list [German](cuisine) [restaurants](object_type:restaurant)
    - do you have any [mexican](cuisine) [restaurants](object_type:restaurant)?
    - do you know the [price range](attribute:price-range) of [that one](mention)?
    - what [cuisine](attribute) is [it](mention)?
    - do you know what [cuisine](attribute) the [last one](mention:LAST) has?
    - does the [first one](mention:1) have [outside seating](attribute:outside-seating)?
    - what is the [price range](attribute:price-range) of [Berlin Burrito Company](restaurant)?
    - what about [I due forni](restaurant)?
    - can you tell me the [price range](attribute) of [that restaurant](mention)?
    - what [cuisine](attribute) do [they](mention) have?
     ...

The above example just shows examples related to the restaurant domain.
You should add examples for every object type that exists in your knowledge base to the same ``query_knowledge_base`` intent.

In addition to adding a variety of training examples for each query type,
you need to specify the and annotate the following entities in your training examples:

- ``object_type``: Whenever a training example references a specific object type from your knowledge base, the object type should
  be marked as an entity. Use :ref:`synonyms <entity_synonyms>` to map e.g. ``restaurants`` to ``restaurant``, the correct
  object type listed as a key in the knowledge base.
- ``mention``: If the user refers to an object via "the first one", "that one", or "it", you should mark those terms
  as ``mention``. We also use synonyms to map some of the mentions to symbols. You can learn about that
  in :ref:`resolving mentions <resolve_mentions>`.
- ``attribute``: All attribute names defined in your knowledge base should be identified as ``attribute`` in the
  NLU data. Again, use synonyms to map variations of an attribute name to the one used in the
  knowledge base.

Remember to add those entities to your domain file (as entities and slots):

.. code-block:: yaml

    entities:
      - object_type
      - mention
      - attribute

    slots:
      object_type:
        type: unfeaturized
      mention:
        type: unfeaturized
      attribute:
        type: unfeaturized


.. _create_action_query_knowledge_base:


Create an Action to Query your Knowledge Base
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To create your own knowledge base action, you need to inherit ``ActionQueryKnowledgeBase`` and pass the knowledge
base to the constructor of ``ActionQueryKnowledgeBase``.

.. code-block:: python

    class MyKnowledgeBaseAction(ActionQueryKnowledgeBase):
        def __init__(self):
            knowledge_base = InMemoryKnowledgeBase("data.json")
            super().__init__(knowledge_base)

Whenever you create an ``ActionQueryKnowledgeBase``, you need to pass a ``KnowledgeBase`` to the constructor.
It can be either an ``InMemoryKnowledgeBase`` or your own implementation of a ``KnowledgeBase``
(see :ref:`custom_knowledge_base`).
You can only pull information from one knowledge base, as the usage of multiple knowledge bases at the same time is not supported.

This is the entirety of the code for this action! The name of the action is ``action_query_knowledge_base``.
Don't forget to add it to your domain file:

.. code-block:: yaml

    actions:
    - action_query_knowledge_base

.. note::
   If you overwrite the default action name ``action_query_knowledge_base``, you need to add the following three
   unfeaturized slots to your domain file: ``knowledge_base_objects``, ``knowledge_base_last_object``, and
   ``knowledge_base_last_object_type``.
   The slots are used internally by ``ActionQueryKnowledgeBase``.
   If you keep the default action name, those slots will be automatically added for you.

You also need to make sure to add a story to your stories file that includes the intent ``query_knowledge_base`` and
the action ``action_query_knowledge_base``. For example:

.. code-block:: md

    ## Happy Path
    * greet
      - utter_greet
    * query_knowledge_base
      - action_query_knowledge_base
    * goodbye
      - utter_goodbye

The last thing you need to do is to define the template ``utter_ask_rephrase`` in your domain file.
If the action doesn't know how to handle the user's request, it will use this template to ask the user to rephrase.
For example, add the following templates to your domain file:

.. code-block:: md

  utter_ask_rephrase:
  - text: "Sorry, I'm not sure I understand. Could you rephrase it?"
  - text: "Could you please rephrase your message? I didn't quite get that."

After adding all the relevant pieces, the action is now able to query the knowledge base.

How It Works
------------

``ActionQueryKnowledgeBase`` looks at both the entities that were picked up in the request as well as the
previously set slots to decide what to query for.

Query the Knowledge Base for Objects
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In order to query the knowledge base for any kind of object, the user's request needs to include the object type.
Let's look at an example:

    `Can you please name some restaurants?`

This question includes the object type of interest: "restaurant."
The bot needs to pick up on this entity in order to formulate a query -- otherwise the action would not know what objects the user is interested in.

When the user says something like:

    `What Italian restaurant options in Berlin do I have?`

The user wants to obtain a list of restaurants that (1) have Italian cuisine and (2) are located in
Berlin. If the NER detects those attributes in the request of the user, the action will use those to filter the
restaurants found in the knowledge base.

In order for the bot to detect these attributes, you need to mark "Italian" and "Berlin" as entities in the NLU data:

.. code-block:: md

    What [Italian](cuisine) [restaurant](object_type) options in [Berlin](city) do I have?.

The names of the attributes, "cuisine" and "city," should be equal to the ones used in the knowledge base.
You also need to add those as entities and slots to the domain file.

Query the Knowledge Base for an Attribute of an Object
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If the user wants to obtain specific information about an object, the request should include both the object and
attribute of interest.
For example, if the user asks something like:

    `What is the cuisine of Berlin Burrito Company?`

The user wants to obtain the "cuisine" (attribute of interest) for the restaurant "Berlin Burrito Company" (object of
interest).

The attribute and object of interest should be marked as entities in the NLU training data:

.. code-block:: md

    What is the [cuisine](attribute) of [Berlin Burrito Company](restaurant)?

Make sure to add the object type, "restaurant," to the domain file as entity and slot.


.. _resolve_mentions:

Resolve Mentions
~~~~~~~~~~~~~~~~

Following along from the above example, users may not always refer to restaurants by their names.
Users can either refer to the object of interest by its name, e.g. "Berlin Burrito Company" (representation string
of the object), or they may refer to a previously listed object via a mention, for example:

    `What is the cuisine of the second restaurant you mentioned?`

Our action is able to resolve these mentions to the actual object in the knowledge base.
More specifically, it can resolve two mention types: (1) ordinal mentions, such as "the first one", and (2)
mentions such as "it" or "that one".

**Ordinal Mentions**

When a user refers to an object by its position in a list, it is called an ordinal mention. Here's an example:

- User: `What restaurants in Berlin do you know?`
- Bot: `Found the following objects of type 'restaurant':  1: I due forni  2: PastaBar  3: Berlin Burrito Company`
- User: `Does the first one have outside seating?`

The user referred to "I due forni" by the term "the first one".
Other ordinal mentions might include "the second one," "the last one," "any," or "3".

Ordinal mentions are typically used when a list of objects was presented to the user.
To resolve those mentions to the actual object, we use an ordinal mention mapping which is set in the
``KnowledgeBase`` class.
The default mapping looks like:

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
          "ANY": lambda l: random.choice(l),
          "LAST": lambda l: l[-1],
      }

The ordinal mention mapping maps a string, such as "1", to the object in a list, e.g. ``lambda l: l[0]``, meaning the
object at index ``0``.

As the ordinal mention mapping does not, for example, include an entry for "the first one",
it is important that you use :ref:`entity_synonyms` to map "the first one" in your NLU data to "1":

.. code-block:: md

    Does the [first one](mention:1) have [outside seating](attribute:outside-seating)?

The NER detects "first one" as a ``mention`` entity, but puts "1" into the ``mention`` slot.
Thus, our action can take the ``mention`` slot together with the ordinal mention mapping to resolve "first one" to
the actual object "I due forni".

You can overwrite the ordinal mention mapping by calling the function ``set_ordinal_mention_mapping()`` on your
``KnowledgeBase`` implementation (see :ref:`customize_in_memory_knowledge_base`).

**Other Mentions**

Take a look at the following conversation:

- User: `What is the cuisine of PastaBar?`
- Bot: `PastaBar has an Italian cuisine.`
- User: `Does it have wifi?`
- Bot: `Yes.`
- User: `Can you give me an address?`

In the question "Does it have wifi?", the user refers to "PastaBar" by the word "it".
If the NER detected "it" as the entity ``mention``, the knowledge base action would resolve it to the last mentioned
object in the conversation, "PastaBar".

In the next input, the user refers indirectly to the object "PastaBar" instead of mentioning it explicitly.
The knowledge base action would detect that the user wants to obtain the value of a specific attribute, in this case, the address.
If no mention or object was detected by the NER, the action assumes the user is referring to the most recently
mentioned object, "PastaBar".

You can disable this behaviour by setting ``use_last_object_mention`` to ``False`` when initializing the action.


Customization
-------------

Customizing ``ActionQueryKnowledgeBase``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You can overwrite the following two functions of ``ActionQueryKnowledgeBase`` if you'd like to customize what the bot
says to the user:

- ``utter_objects()``
- ``utter_attribute_value()``

``utter_objects()`` is used when the user has requested a list of objects.
Once the bot has retrieved the objects from the knowledge base, it will respond to the user by default with a message, formatted like:

    `Found the following objects of type 'restaurant':`
    `1: I due forni`
    `2: PastaBar`
    `3: Berlin Burrito Company`

Or, if no objects are found,

    `I could not find any objects of type 'restaurant'.`

If you want to change the utterance format, you can overwrite the method ``utter_objects()`` in your action.

The function ``utter_attribute_value()`` determines what the bot utters when the user is asking for specific information about
an object.

If the attribute of interest was found in the knowledge base, the bot will respond with the following utterance:

    `'Berlin Burrito Company' has the value 'Mexican' for attribute 'cuisine'.`

If no value for the requested attribute was found, the bot will respond with

    `Did not find a valid value for attribute 'cuisine' for object 'Berlin Burrito Company'.`

If you want to change the bot utterance, you can overwrite the method ``utter_attribute_value()``.

.. note::
   There is a `tutorial <https://blog.rasa.com/integrating-rasa-with-knowledge-bases/>`_ on our blog about
   how to use knowledge bases in custom actions. The tutorial explains the implementation behind
   ``ActionQueryKnowledgeBase`` in detail.


Creating Your Own Knowledge Base Actions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``ActionQueryKnowledgeBase`` should allow you to easily get started with integrating knowledge bases into your actions.
However, the action can only handle two kind of user requests:

- the user wants to get a list of objects from the knowledge base
- the user wants to get the value of an attribute for a specific object

The action is not able to compare objects or consider relations between objects in your knowledge base.
Furthermore, resolving any mention to the last mentioned object in the conversation might not always be optimal.

If you want to tackle more complex use cases, you can write your own custom action.
We added some helper functions to ``rasa_sdk.knowledge_base.utils``
(`link to code <https://github.com/RasaHQ/rasa-sdk/tree/master/rasa_sdk/knowledge_base/>`_ )
to help you when implement your own solution.
We recommend using ``KnowledgeBase`` interface so that you can still use the ``ActionQueryKnowledgeBase``
alongside your new custom action.

If you write a knowledge base action that tackles one of the above use cases or a new one, be sure to tell us about
it on the `forum <https://forum.rasa.com>`_!


.. _customize_in_memory_knowledge_base:

Customizing the ``InMemoryKnowledgeBase``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The class ``InMemoryKnowledgeBase`` inherits ``KnowledgeBase``.
You can customize your ``InMemoryKnowledgeBase`` by overwriting the following functions:

- ``get_key_attribute_of_object()``: To keep track of what object the user was talking about last, we store the value
  of the key attribute in a specific slot. Every object should have a key attribute that is unique,
  similar to the primary key in a relational database. By default, the name of the key attribute for every object type
  is set to ``id``. You can overwrite the name of the key attribute for a specific object type by calling
  ``set_key_attribute_of_object()``.
- ``get_representation_function_of_object()``: Let's focus on the following restaurant:

  .. code-block:: json

      {
          "id": 0,
          "name": "Donath",
          "cuisine": "Italian",
          "outside-seating": true,
          "price-range": "mid-range"
      }

  When the user asks the bot to list any Italian restaurant, it doesn't need all of the details of the restaurant.
  Instead, you want to provide a meaningful name that identifies the restaurant -- in most cases, the name of the object will do.
  The function ``get_representation_function_of_object()`` returns a lambda function that maps the
  above restaurant object to its name.

  .. code-block:: python

      lambda obj: obj["name"]

  This function is used whenever the bot is talking about a specific object, so that the user is presented a meaningful
  name for the object.

  By default, the lambda function returns the value of the ``"name"`` attribute of the object.
  If your object does not have a ``"name"`` attribute , or the ``"name"`` of an object is
  ambiguous, you should set a new lambda function for that object type by calling
  ``set_representation_function_of_object()``.
- ``set_ordinal_mention_mapping()``: The ordinal mention mapping is needed to resolve an ordinal mention, such as
  "second one," to an object in a list. By default, the ordinal mention mapping looks like this:

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
          "ANY": lambda l: random.choice(l),
          "LAST": lambda l: l[-1],
      }

  You can overwrite it by calling the function ``set_ordinal_mention_mapping()``.
  If you want to learn more about how this mapping is used, check out :ref:`resolve_mentions`.


See the `example bot <https://github.com/RasaHQ/rasa/blob/master/examples/knowledgebasebot/actions.py>`_ for an
example implementation of an ``InMemoryKnowledgeBase`` that uses the method ``set_representation_function_of_object()``
to overwrite the default representation of the object type "hotel."
The implementation of the ``InMemoryKnowledgeBase`` itself can be found in the
`rasa-sdk <https://github.com/RasaHQ/rasa-sdk/tree/master/rasa_sdk/knowledge_base/>`_ package.


.. _custom_knowledge_base:

Creating Your Own Knowledge Base
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you have more data or if you want to use a more complex data structure that, for example, involves relations between
different objects, you can create your own knowledge base implementation.
Just inherit ``KnowledgeBase`` and implement the methods ``get_objects()``, ``get_object()``, and
``get_attributes_of_object()``. The `knowledge base code <https://github.com/RasaHQ/rasa-sdk/tree/master/rasa_sdk/knowledge_base/>`_
provides more information on what those methods should do.

You can also customize your knowledge base further, by adapting the methods mentioned in the section
:ref:`customize_in_memory_knowledge_base`.

.. note::
   We wrote a `blog post <https://blog.rasa.com/set-up-a-knowledge-base-to-encode-domain-knowledge-for-rasa/>`_
   that explains how you can set up your own knowledge base.
