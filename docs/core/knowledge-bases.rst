:desc: Leverage information from knowledge bases inside conversations using ActionQueryKnowledgeBase
       in open source bot framework Rasa.

.. _knowledge_base_actions:

Knowledge Base Actions
======================

.. edit-link::

.. warning::
   This feature is experimental.
   We introduce experimental features to get feedback on implementations, so do please try it out.
   However, the functionality might be changed or removed in the future.
   If you have feedback (positive or negative) please comment in the `community forum <https://forum.rasa.com>`_.

.. contents::
   :local:

Knowledge base actions enable you to handle the following kind of dialogues:

.. image:: ../_static/images/knowledge-base-example.png

As you can see users might want to obtain detailed information about certain objects, such as restaurants or hotels,
during a conversation.
They want to obtain details, such as, if a restaurant has outside seating or how expensive the restaurant is.
In order to answer those user requests domain knowledge is needed.
Hard-coding the information would not help as the information are subject to change.
Additionally, users do not only refer to objects by their names, but also use terms, such as "the first one" or "it",
to refer to a specific restaurant.
We need to keep track of what the user spoke about in order to resolve mentions, such "the first one" or "it", to
the correct object.

To handle the above challenges, we recommend that you create a custom action that inherits from
``ActionQueryKnowledgeBase``.
This is a single actions which contains the logic to query a knowledge base for objects and their attributes.
When a restaurant is mentioned indirectly, for example using a phrase like "the first one" or "that restaurant",
this action is able to figure out which restaurant the user is referring to.
You can find a complete example in ``examples/knowledgebasebot``.


Using `ActionQueryKnowledgeBase`
--------------------------------

.. _create_knowledge_base:

Create a Knowledge Base
~~~~~~~~~~~~~~~~~~~~~~~

The data you will use to answer the user's request comes from a knowledge base.
A knowledge base can be used to store complex data structures.
We suggest you get started by using the ``InMemoryKnowledgeBase``.
Once you want to start working with a large amount of data, you can switch to a custom knowledge base
(see :ref:`custom_knowledge_base`).
To initialize an ``InMemoryKnowledgeBase`` you need to provide the data in a json file.

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
The json structure should contain a key for every object type, i.e. "restaurant" and "hotel".
Every object type maps to a list of objects.

Once the data are defined in a json file, called, for example, ``data.json``, you can create your
``InMemoryKnowledgeBase``:

.. code-block:: python

    knowledge_base = InMemoryKnowledgeBase("data.json")

Every object in your knowledge base should have "name" and "id" field.
If that is not the case, please read the section :ref:`customize_in_memory_knowledge_base`.


Defining the NLU Data
~~~~~~~~~~~~~~~~~~~~~

In this section

- we are going to introduce a new intent, ``query_knowledge_base``.
- we are going to annotate ``mention`` entities so that our model detects indirect mentions of objects like "the
  first one".
- we will use synonyms (:ref:`entity_synonyms`) extensively.

To be able to understand that the user wants to retrieve some information from the knowledge base, you need to define
a new intent, for example, called ``query_knowledge_base``.
The intent should contain all kind of user requests.

Let's look at an example:

.. code-block:: md

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
    - does the [first one](mention:1) have [outside seating](attribute:outside-seating)?
    - what is the [price range](attribute:price-range) of [Berlin Burrito Company](restaurant)?
    - what is with [I due forni](restaurant)?
     ...

The above example just shows examples related to the restaurant domain.
You should add examples for every object type that exists in your knowledge base.

All user requests can be divided into two categories:
(1) The user wants to obtain a list of objects of a specific type or (2) the user wants to know about a certain
attribute of an object.
The ``ActionQueryKnowledgeBase`` can handle both of those requests.
Other requests, such as comparison between objects, are currently not supported.

Another thing you may have noticed is, that we marked different kind of entities in the NLU data.
If you want to use ``ActionQueryKnowledgeBase``, you need to specify the following entities:

- ``object_type``: Whenever the user is talking about a specific object type from your knowledge base, the type should
  be marked as entity in our NLU data. Use :ref:`entity_synonyms` to map, for example, "restaurants" to the correct
  object type listed in the knowledge base, e.g. "restaurant".
- ``mention``: If the user refers to an object via "the first one", "that one", or "it", you should mark those terms
  as ``mention``. We also use :ref:`entity_synonyms` to map some of the mentions to symbols. You can learn about that
  in section :ref:`resolve_mentions`.
- ``attribute``: All attribute names defined in your knowledge base should be identified as ``attribute`` in the
  NLU data. Again, use :ref:`entity_synonyms` to map variations of an attribute name to the one used in the
  knowledge base.

Remember to add those entities to your domain file (as entities and slots):

.. code-block:: md

    entities:
      - object_type
      - mention
      - attribute

    slots:
      object_type:
        type: text
      mention:
        type: text
      attribute:
        type: text


.. _create_action_query_knowledge_base:


Create an Action to query your Knowledge Base
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Whenever you create an ``ActionQueryKnowledgeBase``, you need to pass a ``KnowledgeBase`` to the constructor.
It can be either an ``InMemoryKnowledgeBase`` or your own implementation of a ``KnowledgeBase``
(see :ref:`create_knowledge_base`).
However, you can just use one knowledge base.
The usage of multiple knowledge bases at the same time is not supported.

To create your own knowledge base action, you need to inherit ``ActionQueryKnowledgeBase`` and pass the knowledge
base to the constructor of ``ActionQueryKnowledgeBase``.

.. code-block:: python

    class MyKnowledgeBaseAction(ActionQueryKnowledgeBase):
        def __init__(self):
            knowledge_base = InMemoryKnowledgeBase("data.json")
            super().__init__(knowledge_base)

The name of the action is ``action_query_knowledge_base``.
Don't forget to add it to your domain file.

.. code-block:: md

    actions:
    - action_query_knowledge_base

.. note::
   If you overwrite the default action name ``action_query_knowledge_base``, you need to add the following three
   unfeaturized slots to your domain file: ``knowledge_base_objects``, ``knowledge_base_last_object``, and
   ``knowledge_base_last_object_type``.
   The slots are used internally by ``ActionQueryKnowledgeBase``.
   If you keep the default action name, those slots will be automatically added for you.

You also need to make sure, to add a story to your stories file that includes the intent ``query_knowledge_base`` and
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
If the action does not know how to handle the request of the user, it will use the template to tell the user, that
it is lost and the user should rephrase its request.
You could, for example, add the following to your domain file:

.. code-block:: md

  utter_ask_rephrase:
  - text: "Sorry, I'm not sure I understand. Can you rephrase?"
  - text: "Can you please rephrase? I did not got that."

You don't need to do anything else.
The action is now able to query the knowledge base.

How it works
------------

In general the ``ActionQueryKnowledgeBase`` looks at the entities that were picked up in the request and the
previous set slots to decide what to query for.

Query the Knowledge Base for Objects
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In order to query the knowledge base for any kind of objects, the user's request needs to include the object type.
Let's look at an example:

    `Can you please name some restaurants?`

The question includes the object type of interest: "restaurant".
If the request would not contain the type of interest, the action would not know what objects the user is interested in.
The action would not be able to formulate a query.
What when the user says something like:

    `What Italian restaurant options in Berlin do I have?`

In this example the user want to obtain a list of restaurants that (1) have an Italian cuisine and (2) are located in
Berlin.
In order to filter the objects in the knowledge base, you need to mark "Italian" and "Berlin" as entities.
E.g.

.. code-block:: md

    What [Italian](cuisine) [restaurant](object_type) options in [Berlin](city) do I have?.

The names of the attributes, e.g. "cuisine" and "city", should be equal to the ones used in the knowledge base.
You also need to add those as entities and slots to the domain file.
If the NER detects those attributes in the request of the user, the action will use those for filter the
restaurants found in the knowledge base.


Query the Knowledge Base for an Attribute of an Object
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If the user wants to obtain a detail of a certain object, the request of the user should include the object and
attribute of interest.
For example, if the user asks something like

    `What is the cuisine of Berlin Burrito Company?`

the user wants to obtain the "cuisine" (attribute of interest) for the restaurant "Berlin Burrito Company" (object of
interest).

However, users do not always refer to restaurants by their names.
Users can either refer to the object of interest by its name, e.g. "Berlin Burrito Company" (representation string
of the object), or they refer to a previously listed object via a mention, e.g.

    `What is the cuisine of the second restaurant you just mentioned?`

To learn more about how we resolve those mentions to the actual object in the knowledge base, go to section
:ref:`resolve_mentions`.

The attribute and object of interest should be marked as entities in the NLU training data, e.g.

.. code-block:: md

    What is the [cuisine](attribute) of [Berlin Burrito Company](restaurant)?

Make sure to add the object type, e.g. "restaurant", to the domain file as entity and slot.

.. _resolve_mentions:

Resolve Mentions
~~~~~~~~~~~~~~~~

Looking at the example from the beginning, we saw that users refer to previously mentioned objects during a conversation
in different ways.
Our action is able to (1) resolve ordinal mentions, such as "the first one", to the actual object and (2) resolve
mentions, such as "it" or "that one", to the last mentioned object in the conversation.
Let's take a closer look.

**Ordinal Mentions**

If the user refers to an object by its position in a list, we talk about ordinal mentions.
Let's look at an example conversation:

- User: `What restaurants in Berlin do you know?`
- Bot: `Found the following objects of type 'restaurant':  1: I due forni  2: PastaBar  3: Berlin Burrito Company`
- User: `Does the first one have outside seating?`

The user referred to "I due forni" by the term "the first one".
Other ordinal mentions are, for example:

- `the second one`
- `the last one`
- `any`
- `3`

Ordinal mentions are typically used when a list of objects was presented to the user.
To resolve those mentions to the actual object, we use an ordinal mention mapping which is set in the
``KnowledgeBase`` class.
The default mapping looks like the following:

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

The ordinal mention mapping maps a string, such as "1", to the object in a list, e.g. ``lambda l: l[0]``.
You can overwrite the ordinal mention mapping by calling the function ``set_ordinal_mention_mapping()`` on your
``KnowledgeBase`` implementation (see :ref:`customize_in_memory_knowledge_base`).
As the ordinal mention mapping does not, for example, include an entry for "the first one".
It is important that you use :ref:`entity_synonyms` to map "the first one" in your NLU data to "1".
For example,

.. code-block:: md

    Does the [first one](mention:1) have [outside seating](attribute:outside-seating)?

maps "first one" via a synonym to "1".
The NER detects "first one" as ``mention`` entity, but puts "1" into the ``mention`` slot.
Thus, our action can take the ``mention`` slot together with the ordinal mention mapping to resolve "first one" to
the actual object "I due forni".

Other Mentions
~~~~~~~~~~~~~~
Take a look at the following conversation:

- User: `What is the cuisine of PastaBar?`
- Bot: `PastaBar has an Italian cuisine.`
- User: `Does it have wifi?`
- Bot: `Yes.`
- User: `Can you give me an address?`

In the second utterance of the user, the user refers to "PastaBar" by the word "it".
If the NER detected "it" as the entity ``mention``, the knowledge base action would resolve it to the last mentioned
object in the conversation, e.g. "PastaBar".
In the next utterance of the user, the user refers indirectly to the object "PastaBar".
However, the user does not mention "PastaBar" explicitly.
The knowledge base action would detect that the user wants to obtain the value of a specific attribute.
If no mention or object could be detected by the NER, the action just assumes the user is talking about he last
mentioned object, e.g. "PastaBar".
You can disable this behaviour by setting ``use_last_object_mention`` to ``False`` when initializing the action.


Customization
-------------

Customize your `ActionQueryKnowledgeBase`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You can overwrite the following two functions of `ActionQueryKnowledgeBase`:

- ``utter_objects()``
- ``utter_attribute_value()``

``utter_objects()`` is used when the user requested the bot to list some objects.
Once the bot retrieved some objects from the knowledge base, it will response to the user, for example, with

    `Found the following objects of type 'restaurant':`
    `1: I due forni`
    `2: PastaBar`
    `3: Berlin Burrito Company`

Or if no entities could be found

    `I could not find any objects of type 'restaurant'.`

If you want to change the utterance of the bot, you can overwrite the method ``utter_objects()`` in your action.

The function ``utter_attribute_value()`` determines what the bot utters when the user is asking for a detail of
an object.
If the attribute of interest was found in the knowledge base, the bot will response with the following utterance:

    `'Berlin Burrito Company' has the value 'Mexican' for attribute 'cuisine'.`

If no value for the requested attribute was found, the bot will response with

    `Did not find a valid value for attribute 'cuisine' for object 'Berlin Burrito Company'.`

If you want to change the utterance of the bot, you can overwrite the method ``utter_attribute_value()``.

.. note::
   There is a tutorial `here <https://blog.rasa.com/integrating-rasa-with-knowledge-bases/>`_ about how to use
   knowledge bases in custom actions. The tutorial will explain in detail the implementation behind
   ``ActionQueryKnowledgeBase``.


Creating your own Knowledge Base Actions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``ActionQueryKnowledgeBase`` should allow you to get easily started with using a knowledge base for Rasa.
However, the action can only handle two kind of user requests:

- the user wants to get a list of objects from the knowledge base or
- the user wants to get the value of an attribute for a specific object

The action, for example, is not able to compare objects or consider relations between objects in your knowledge base.
Furthermore, resolving any mention to the last mentioned object in the conversation, might not always be optimal.
If you want to tackle more complex use cases, you can write your own custom action.
We added some helper function to ``rasa_sdk.knowledge_base.utils`` that might help you when implementing your own
solution.
We recommend to use the ``KnowledgeBase`` interface, so that you can still use the ``ActionQueryKnowledgeBase``
alongside your new custom action.


.. _customize_in_memory_knowledge_base:

Customize your `InMemoryKnowledgeBase`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The class ``InMemoryKnowledgeBase`` inherits ``KnowledgeBase``.
You can customize your ``InMemoryKnowledgeBase`` by overwriting the following functions:

- ``get_key_attribute_of_object``: To keep track of what object the user was talking about last, we store the value
  of the key attribute in a specific slot. Every object should have a key attribute that is unique, i.e.
  similar to the primary key in a relation database. By default the name of the key attribute for every object type
  is set to "id". You can overwrite the name of the key attribute for a specific object type by calling
  ``set_key_attribute_of_object()``.
- ``get_representation_function_of_object``: Let's focus on the following restaurant:

  .. code-block:: json

      {
          "id": 0,
          "name": "Donath",
          "cuisine": "Italian",
          "outside-seating": true,
          "price-range": "mid-range"
      }

  When the user is asking to list any Italian restaurant, you don't want to confront the user with all details of that
  restaurant. You want to provide a meaningful name that identifies the restaurant. Most likely you would use
  just the name of the restaurant to speak about it.
  Thus, the function ``get_representation_function_of_object`` returns a lambda function that maps, for example, the
  above restaurant object to its name.

  .. code-block:: python

      lambda obj: obj["name"]

  This function is used whenever the bot is talking about a specific object, so that the user is given a meaningful
  name and he knows what exactly the bot is talking about.
  By default the lambda function is set to ``lambda obj: obj["name"]``. So, it returns the value of the attribute
  "name" of the object. If your object does not have an attribute "name", or the "name" of an object might be
  ambiguous, you should set a new lambda function for that object type by calling
  ``set_representation_function_of_object()``.
- ``set_ordinal_mention_mapping``: The ordinal mention mapping is needed to resolve an ordinal mention to an object
  in a list. For example, if the bot listed a few restaurants in Berlin, and the user then asked

    `Does the second one have outside seating?`

  you need to resolve "second one" to the correct object the bot listed before. Per
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
  If you want to learn more about the usage of the mapping, go to section :ref:`resolve_mentions`.


.. _custom_knowledge_base:

Creating your own Knowledge Base
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you have more data or if you want to use a more complex data structure that, for example, involves relations between
different objects, you can create your own knowledge base implementation.
Just inherit ``KnowledgeBase`` and implement the methods ``get_objects()``, ``get_object()``, and
``get_attributes_of_object()``.
You can also customize your knowledge base further, for example, by adapting the methods mentioned in the section
:ref:`customize_in_memory_knowledge_base`.

.. note::
   We wrote a `blog post <https://blog.rasa.com/set-up-a-knowledge-base-to-encode-domain-knowledge-for-rasa/>`_
   that explains how you can set up your own knowledge base.
