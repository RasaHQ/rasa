:desc: Leverage information from knowledge bases inside conversations using KnowledgeBaseActions
       in open source bot framework Rasa.

.. _knowledge_bases:

Knowledge Base Actions
======================

.. edit-link::

.. note::
   There is a tutorial `here <https://blog.rasa.com/integrating-rasa-with-knowledge-bases/>`_ about how to use
   knowledge bases in custom actions.

.. contents::
   :local:

A lot of users want to obtain detailed information about certain objects, such as restaurants, during a conversation.
In order to answer those user requests domain knowledge is needed.
Hard-coding the information would not help as the information are subject to change.
Additionally, users do not only refer to objects by their names, but also use terms, such as "the first one" or "that
restaurant", to refer to a specific object.
Objects need to be recognised and reused at a later point in the conversation.

To handle the above challenges, we recommend that you create a ``ActionQueryKnowledgeBase``.
This is a single actions which contains the logic to query a knowledge base for objects and their attributes.
The action is also able to resolve certain mention of objects, such as ``the first one`` or ``that restaurant``.
You can find a complete example in ``examples/knowledge_base_bot``.

Create a ActionQueryKnowledgeBase
---------------------------------

Whenever you create a ``ActionQueryKnowledgeBase``, you need to pass a ``KnowledgeBase`` to the constructor.
A knowledge base can be used to store complex data structures.
If you just have some data points that fit into memory, you can use our ``InMemoryKnowledgeBase`` implementation.
To initialize a ``InMemoryKnowledgeBase`` you need to provide the data and the schema in form of a python dictionary.
The schema and data need to follow a specific data structure.

Let's take a look at an example:

.. code-block:: python

    schema = {
        "restaurant": {
            "attributes": [],
            "key": "name",
            "representation": lambda e: e["name"],
        },
        "hotel": {
            "attributes": [],
            "key": "name",
            "representation": lambda e: e["name"] + " (" + e["city"] + ")",
        },
    }

    data = {
        "restaurant": [
        ],
        "hotel": [
        ],
    }

    knowledge_base = InMemoryKnowledgeBase(schema, data)

The example has two different object types: ``restaurant`` and ``hotel``.
The data dictionary should contain a list of objects for each object type.
The schema dictionary must contain the keys ``attributes``, ``key``, and ``representation`` for all object types.
``attributes`` defines the attributes the user can filter for.
``key`` is the key attribute of that object.
The value of the key attribute of every object should be unique.
And ``representation`` maps to a lamdba function that is used to map the object to a string representation.
This function is used whenever an object is outputted by the bot.

Once the schema and the data are defined, you can create a ``InMemoryKnowledgeBase``.
If you have more data you can also create your own knowledge base implementation.
Just inherit ``KnowledgeBase`` and implement the methods ``get_objects()`` and ``get_attribute_of()``.

.. note::
   There is a tutorial `here <https://blog.rasa.com/set-up-a-knowledge-base-to-encode-domain-knowledge-for-rasa/>`_
   that explains how you can set up your own knowledge base.

As soon as you defined your knowledge base, you can actually create the ``ActionQueryKnowledgeBase``.

.. code-block:: python

    class MyKnowledgeBaseAction(ActionQueryKnowledgeBase):
        def __init__(self):
            knowledge_base = InMemoryKnowledgeBase(schema, data)
            super().__init__(knowledge_base)

You don't need to do anything else.
The action is already able to query the knowledge base.
The name of the action is ``action_query_knowledge_base``.
Don't forget to add it to your domain file.

Defining the NLU Data
---------------------

To be able that the user wants to retrieve some information from the knowledge base, you need to define a new intent,
e.g. ``query_knowledge_base``.
The intent should contain all kind of user requests.

Let's look at an example:

.. code-block:: yaml

    ## intent:query_knowledge_base
    - what [restaurants](object_type:restaurant) can you recommend?
    - list some [restaurants](object_type:restaurant)
    - can you show me some [restaurant](object_type:restaurant) options
    - does the [first](mention:1) one has [wifi](attribute)?
    - what [cuisine](attribute) is [it](mention)?
    - does the [last](mention:LAST) one offer [breakfast](attribute:breakfast-included)?
    - do you know the [cuisine](attribute) of [that one](mention)?
    - do you have any [mexican](cuisine) [restaurants](object_type:restaurant)?
    - can you name some [restaurants](object_type:restaurant), please?
    - do you know what [cuisine](attribute) the [last one](mention:LAST) has?
    - does [PastaBar](restaurant) have [wifi](attribute)?
    - what is the [cuisine](attribute) of [Berlin Burrito Company](restaurant)?
    - what is with [I due forni](restaurant)?
     ...

The above examples just cover the restaurant objects.
You should add examples for every object type that exists in your knowledge base.

As you can see, all requests can be divided into two categories: The user wants to obtain a list of objects from a
specific type or the user wants to know about a certain attribute of an object.
The ``ActionQueryKnowledgeBase`` can handle both of those requests.
Other requests, such as comparison between objects, are currently not supoorted.

We marked a few entities in the NLU data.
If you want to use ``ActionQueryKnowledgeBase``, you need to specify the following entities:

- ``object_type``: Whenever the user is talking about a specific object type from your knowledge base, the type should
be extracted by the NER. Use synonyms (TODO: link) to map, for example, "restaurants" to the correct object type listed
in the knowledge base, e.g. "restaurant".
- ``mention``: If the user refers to an object via "the first one", "that one", or "it", you should mark those terms
as ``mention``. We also use synonyms to map some of the mentions to symbols. More on that in TODO-link.
- ``attribute``: All attribute names defined in your knowledge base should be marked in the NLU data. Again, use
synonyms to map variations of an attribute name to the one used in the knowledge base.

Don't forget to add those entities to your domain file once as entities and once as slots.


Query the Knowledge Base for Objects
------------------------------------


Query the Knowledge Base for an Attribute of an Object
------------------------------------------------------


Resolve Mentions
----------------


Limitations of ActionQueryKnowledgeBase
---------------------------------------


