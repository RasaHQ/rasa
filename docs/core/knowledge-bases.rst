:desc: Leverage information from knowledge bases inside conversations using ActionQueryKnowledgeBase a
       in open source bot framework Rasa. a
 a
.. _knowledge_base_actions: a
 a
Knowledge Base Actions a
====================== a
 a
.. edit-link:: a
 a
.. warning:: a
   This feature is experimental. a
   We introduce experimental features to get feedback from our community, so we encourage you to try it out! a
   However, the functionality might be changed or removed in the future. a
   If you have feedback (positive or negative) please share it with us on the `forum <https://forum.rasa.com>`_. a
 a
.. contents:: a
   :local: a
 a
Knowledge base actions enable you to handle the following kind of conversations: a
 a
.. image:: ../_static/images/knowledge-base-example.png a
 a
A common problem in conversational AI is that users do not only refer to certain objects by their names, a
but also use reference terms such as "the first one" or "it". a
We need to keep track of the information that was presented to resolve these mentions to a
the correct object. a
 a
In addition, users may want to obtain detailed information about objects during a conversation -- a
for example, whether a restaurant has outside seating, or how expensive it is. a
In order to respond to those user requests, knowledge about the restaurant domain is needed. a
Since the information is subject to change, hard-coding the information isn't the solution. a
 a
 a
To handle the above challenges, Rasa can be integrated with knowledge bases. To use this integration, you can create a a
custom action that inherits from ``ActionQueryKnowledgeBase``, a pre-written custom action that contains a
the logic to query a knowledge base for objects and their attributes. a
 a
You can find a complete example in ``examples/knowledgebasebot`` a
(`knowledge base bot <https://github.com/RasaHQ/rasa/tree/master/examples/knowledgebasebot/>`_), as well as instructions a
for implementing this custom action below. a
 a
 a
Using ``ActionQueryKnowledgeBase`` a
---------------------------------- a
 a
.. _create_knowledge_base: a
 a
Create a Knowledge Base a
~~~~~~~~~~~~~~~~~~~~~~~ a
 a
The data used to answer the user's requests will be stored in a knowledge base. a
A knowledge base can be used to store complex data structures. a
We suggest you get started by using the ``InMemoryKnowledgeBase``. a
Once you want to start working with a large amount of data, you can switch to a custom knowledge base a
(see :ref:`custom_knowledge_base`). a
 a
To initialize an ``InMemoryKnowledgeBase``, you need to provide the data in a json file. a
The following example contains data about restaurants and hotels. a
The json structure should contain a key for every object type, i.e. ``"restaurant"`` and ``"hotel"``. a
Every object type maps to a list of objects -- here we have a list of 3 restaurants and a list of 3 hotels. a
 a
.. code-block:: json a
 a
    { a
        "restaurant": [ a
            { a
                "id": 0, a
                "name": "Donath", a
                "cuisine": "Italian", a
                "outside-seating": true, a
                "price-range": "mid-range" a
            }, a
            { a
                "id": 1, a
                "name": "Berlin Burrito Company", a
                "cuisine": "Mexican", a
                "outside-seating": false, a
                "price-range": "cheap" a
            }, a
            { a
                "id": 2, a
                "name": "I due forni", a
                "cuisine": "Italian", a
                "outside-seating": true, a
                "price-range": "mid-range" a
            } a
        ], a
        "hotel": [ a
            { a
                "id": 0, a
                "name": "Hilton", a
                "price-range": "expensive", a
                "breakfast-included": true, a
                "city": "Berlin", a
                "free-wifi": true, a
                "star-rating": 5, a
                "swimming-pool": true a
            }, a
            { a
                "id": 1, a
                "name": "Hilton", a
                "price-range": "expensive", a
                "breakfast-included": true, a
                "city": "Frankfurt am Main", a
                "free-wifi": true, a
                "star-rating": 4, a
                "swimming-pool": false a
            }, a
            { a
                "id": 2, a
                "name": "B&B", a
                "price-range": "mid-range", a
                "breakfast-included": false, a
                "city": "Berlin", a
                "free-wifi": false, a
                "star-rating": 1, a
                "swimming-pool": false a
            }, a
        ] a
    } a
 a
 a
Once the data is defined in a json file, called, for example, ``data.json``, you will be able use the this data file to create your a
``InMemoryKnowledgeBase``, which will be passed to the action that queries the knowledge base. a
 a
Every object in your knowledge base should have at least the ``"name"`` and ``"id"`` fields to use the default implementation. a
If it doesn't, you'll have to :ref:`customize your InMemoryKnowledgeBase <customize_in_memory_knowledge_base>`. a
 a
 a
Define the NLU Data a
~~~~~~~~~~~~~~~~~~~ a
 a
In this section: a
 a
- we will introduce a new intent, ``query_knowledge_base`` a
- we will to annotate ``mention`` entities so that our model detects indirect mentions of objects like "the a
  first one" a
- we will use :ref:`synonyms <entity_synonyms>` extensively a
 a
For the bot to understand that the user wants to retrieve information from the knowledge base, you need to define a
a new intent. We will call it ``query_knowledge_base``. a
 a
We can split requests that ``ActionQueryKnowledgeBase`` can handle into two categories: a
(1) the user wants to obtain a list of objects of a specific type, or (2) the user wants to know about a certain a
attribute of an object. The intent should contain lots of variations of both of these requests: a
 a
.. code-block:: md a
 a
    ## intent:query_knowledge_base a
    - what [restaurants](object_type:restaurant) can you recommend? a
    - list some [restaurants](object_type:restaurant) a
    - can you name some [restaurants](object_type:restaurant) please? a
    - can you show me some [restaurant](object_type:restaurant) options a
    - list [German](cuisine) [restaurants](object_type:restaurant) a
    - do you have any [mexican](cuisine) [restaurants](object_type:restaurant)? a
    - do you know the [price range](attribute:price-range) of [that one](mention)? a
    - what [cuisine](attribute) is [it](mention)? a
    - do you know what [cuisine](attribute) the [last one](mention:LAST) has? a
    - does the [first one](mention:1) have [outside seating](attribute:outside-seating)? a
    - what is the [price range](attribute:price-range) of [Berlin Burrito Company](restaurant)? a
    - what about [I due forni](restaurant)? a
    - can you tell me the [price range](attribute) of [that restaurant](mention)? a
    - what [cuisine](attribute) do [they](mention) have? a
     ... a
 a
The above example just shows examples related to the restaurant domain. a
You should add examples for every object type that exists in your knowledge base to the same ``query_knowledge_base`` intent. a
 a
In addition to adding a variety of training examples for each query type, a
you need to specify the and annotate the following entities in your training examples: a
 a
- ``object_type``: Whenever a training example references a specific object type from your knowledge base, the object type should a
  be marked as an entity. Use :ref:`synonyms <entity_synonyms>` to map e.g. ``restaurants`` to ``restaurant``, the correct a
  object type listed as a key in the knowledge base. a
- ``mention``: If the user refers to an object via "the first one", "that one", or "it", you should mark those terms a
  as ``mention``. We also use synonyms to map some of the mentions to symbols. You can learn about that a
  in :ref:`resolving mentions <resolve_mentions>`. a
- ``attribute``: All attribute names defined in your knowledge base should be identified as ``attribute`` in the a
  NLU data. Again, use synonyms to map variations of an attribute name to the one used in the a
  knowledge base. a
 a
Remember to add those entities to your domain file (as entities and slots): a
 a
.. code-block:: yaml a
 a
    entities: a
      - object_type a
      - mention a
      - attribute a
 a
    slots: a
      object_type: a
        type: unfeaturized a
      mention: a
        type: unfeaturized a
      attribute: a
        type: unfeaturized a
 a
 a
.. _create_action_query_knowledge_base: a
 a
 a
Create an Action to Query your Knowledge Base a
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ a
 a
To create your own knowledge base action, you need to inherit ``ActionQueryKnowledgeBase`` and pass the knowledge a
base to the constructor of ``ActionQueryKnowledgeBase``. a
 a
.. code-block:: python a
 a
    from rasa_sdk.knowledge_base.storage import InMemoryKnowledgeBase a
    from rasa_sdk.knowledge_base.actions import ActionQueryKnowledgeBase a
 a
    class MyKnowledgeBaseAction(ActionQueryKnowledgeBase): a
        def __init__(self): a
            knowledge_base = InMemoryKnowledgeBase("data.json") a
            super().__init__(knowledge_base) a
 a
Whenever you create an ``ActionQueryKnowledgeBase``, you need to pass a ``KnowledgeBase`` to the constructor. a
It can be either an ``InMemoryKnowledgeBase`` or your own implementation of a ``KnowledgeBase`` a
(see :ref:`custom_knowledge_base`). a
You can only pull information from one knowledge base, as the usage of multiple knowledge bases at the same time is not supported. a
 a
This is the entirety of the code for this action! The name of the action is ``action_query_knowledge_base``. a
Don't forget to add it to your domain file: a
 a
.. code-block:: yaml a
 a
    actions: a
    - action_query_knowledge_base a
 a
.. note:: a
   If you overwrite the default action name ``action_query_knowledge_base``, you need to add the following three a
   unfeaturized slots to your domain file: ``knowledge_base_objects``, ``knowledge_base_last_object``, and a
   ``knowledge_base_last_object_type``. a
   The slots are used internally by ``ActionQueryKnowledgeBase``. a
   If you keep the default action name, those slots will be automatically added for you. a
 a
You also need to make sure to add a story to your stories file that includes the intent ``query_knowledge_base`` and a
the action ``action_query_knowledge_base``. For example: a
 a
.. code-block:: md a
 a
    ## Happy Path a
    * greet a
      - utter_greet a
    * query_knowledge_base a
      - action_query_knowledge_base a
    * goodbye a
      - utter_goodbye a
 a
The last thing you need to do is to define the response ``utter_ask_rephrase`` in your domain file. a
If the action doesn't know how to handle the user's request, it will use this response to ask the user to rephrase. a
For example, add the following responses to your domain file: a
 a
.. code-block:: md a
 a
  utter_ask_rephrase: a
  - text: "Sorry, I'm not sure I understand. Could you rephrase it?" a
  - text: "Could you please rephrase your message? I didn't quite get that." a
 a
After adding all the relevant pieces, the action is now able to query the knowledge base. a
 a
How It Works a
------------ a
 a
``ActionQueryKnowledgeBase`` looks at both the entities that were picked up in the request as well as the a
previously set slots to decide what to query for. a
 a
Query the Knowledge Base for Objects a
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ a
 a
In order to query the knowledge base for any kind of object, the user's request needs to include the object type. a
Let's look at an example: a
 a
    `Can you please name some restaurants?` a
 a
This question includes the object type of interest: "restaurant." a
The bot needs to pick up on this entity in order to formulate a query -- otherwise the action would not know what objects the user is interested in. a
 a
When the user says something like: a
 a
    `What Italian restaurant options in Berlin do I have?` a
 a
The user wants to obtain a list of restaurants that (1) have Italian cuisine and (2) are located in a
Berlin. If the NER detects those attributes in the request of the user, the action will use those to filter the a
restaurants found in the knowledge base. a
 a
In order for the bot to detect these attributes, you need to mark "Italian" and "Berlin" as entities in the NLU data: a
 a
.. code-block:: md a
 a
    What [Italian](cuisine) [restaurant](object_type) options in [Berlin](city) do I have?. a
 a
The names of the attributes, "cuisine" and "city," should be equal to the ones used in the knowledge base. a
You also need to add those as entities and slots to the domain file. a
 a
Query the Knowledge Base for an Attribute of an Object a
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ a
 a
If the user wants to obtain specific information about an object, the request should include both the object and a
attribute of interest. a
For example, if the user asks something like: a
 a
    `What is the cuisine of Berlin Burrito Company?` a
 a
The user wants to obtain the "cuisine" (attribute of interest) for the restaurant "Berlin Burrito Company" (object of a
interest). a
 a
The attribute and object of interest should be marked as entities in the NLU training data: a
 a
.. code-block:: md a
 a
    What is the [cuisine](attribute) of [Berlin Burrito Company](restaurant)? a
 a
Make sure to add the object type, "restaurant," to the domain file as entity and slot. a
 a
 a
.. _resolve_mentions: a
 a
Resolve Mentions a
~~~~~~~~~~~~~~~~ a
 a
Following along from the above example, users may not always refer to restaurants by their names. a
Users can either refer to the object of interest by its name, e.g. "Berlin Burrito Company" (representation string a
of the object), or they may refer to a previously listed object via a mention, for example: a
 a
    `What is the cuisine of the second restaurant you mentioned?` a
 a
Our action is able to resolve these mentions to the actual object in the knowledge base. a
More specifically, it can resolve two mention types: (1) ordinal mentions, such as "the first one", and (2) a
mentions such as "it" or "that one". a
 a
**Ordinal Mentions** a
 a
When a user refers to an object by its position in a list, it is called an ordinal mention. Here's an example: a
 a
- User: `What restaurants in Berlin do you know?` a
- Bot: `Found the following objects of type 'restaurant':  1: I due forni  2: PastaBar  3: Berlin Burrito Company` a
- User: `Does the first one have outside seating?` a
 a
The user referred to "I due forni" by the term "the first one". a
Other ordinal mentions might include "the second one," "the last one," "any," or "3". a
 a
Ordinal mentions are typically used when a list of objects was presented to the user. a
To resolve those mentions to the actual object, we use an ordinal mention mapping which is set in the a
``KnowledgeBase`` class. a
The default mapping looks like: a
 a
  .. code-block:: python a
 a
      { a
          "1": lambda l: l[0], a
          "2": lambda l: l[1], a
          "3": lambda l: l[2], a
          "4": lambda l: l[3], a
          "5": lambda l: l[4], a
          "6": lambda l: l[5], a
          "7": lambda l: l[6], a
          "8": lambda l: l[7], a
          "9": lambda l: l[8], a
          "10": lambda l: l[9], a
          "ANY": lambda l: random.choice(l), a
          "LAST": lambda l: l[-1], a
      } a
 a
The ordinal mention mapping maps a string, such as "1", to the object in a list, e.g. ``lambda l: l[0]``, meaning the a
object at index ``0``. a
 a
As the ordinal mention mapping does not, for example, include an entry for "the first one", a
it is important that you use :ref:`entity_synonyms` to map "the first one" in your NLU data to "1": a
 a
.. code-block:: md a
 a
    Does the [first one](mention:1) have [outside seating](attribute:outside-seating)? a
 a
The NER detects "first one" as a ``mention`` entity, but puts "1" into the ``mention`` slot. a
Thus, our action can take the ``mention`` slot together with the ordinal mention mapping to resolve "first one" to a
the actual object "I due forni". a
 a
You can overwrite the ordinal mention mapping by calling the function ``set_ordinal_mention_mapping()`` on your a
``KnowledgeBase`` implementation (see :ref:`customize_in_memory_knowledge_base`). a
 a
**Other Mentions** a
 a
Take a look at the following conversation: a
 a
- User: `What is the cuisine of PastaBar?` a
- Bot: `PastaBar has an Italian cuisine.` a
- User: `Does it have wifi?` a
- Bot: `Yes.` a
- User: `Can you give me an address?` a
 a
In the question "Does it have wifi?", the user refers to "PastaBar" by the word "it". a
If the NER detected "it" as the entity ``mention``, the knowledge base action would resolve it to the last mentioned a
object in the conversation, "PastaBar". a
 a
In the next input, the user refers indirectly to the object "PastaBar" instead of mentioning it explicitly. a
The knowledge base action would detect that the user wants to obtain the value of a specific attribute, in this case, the address. a
If no mention or object was detected by the NER, the action assumes the user is referring to the most recently a
mentioned object, "PastaBar". a
 a
You can disable this behaviour by setting ``use_last_object_mention`` to ``False`` when initializing the action. a
 a
 a
Customization a
------------- a
 a
Customizing ``ActionQueryKnowledgeBase`` a
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ a
 a
You can overwrite the following two functions of ``ActionQueryKnowledgeBase`` if you'd like to customize what the bot a
says to the user: a
 a
- ``utter_objects()`` a
- ``utter_attribute_value()`` a
 a
``utter_objects()`` is used when the user has requested a list of objects. a
Once the bot has retrieved the objects from the knowledge base, it will respond to the user by default with a message, formatted like: a
 a
    `Found the following objects of type 'restaurant':` a
    `1: I due forni` a
    `2: PastaBar` a
    `3: Berlin Burrito Company` a
 a
Or, if no objects are found, a
 a
    `I could not find any objects of type 'restaurant'.` a
 a
If you want to change the utterance format, you can overwrite the method ``utter_objects()`` in your action. a
 a
The function ``utter_attribute_value()`` determines what the bot utters when the user is asking for specific information about a
an object. a
 a
If the attribute of interest was found in the knowledge base, the bot will respond with the following utterance: a
 a
    `'Berlin Burrito Company' has the value 'Mexican' for attribute 'cuisine'.` a
 a
If no value for the requested attribute was found, the bot will respond with a
 a
    `Did not find a valid value for attribute 'cuisine' for object 'Berlin Burrito Company'.` a
 a
If you want to change the bot utterance, you can overwrite the method ``utter_attribute_value()``. a
 a
.. note:: a
   There is a `tutorial <https://blog.rasa.com/integrating-rasa-with-knowledge-bases/>`_ on our blog about a
   how to use knowledge bases in custom actions. The tutorial explains the implementation behind a
   ``ActionQueryKnowledgeBase`` in detail. a
 a
 a
Creating Your Own Knowledge Base Actions a
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ a
 a
``ActionQueryKnowledgeBase`` should allow you to easily get started with integrating knowledge bases into your actions. a
However, the action can only handle two kind of user requests: a
 a
- the user wants to get a list of objects from the knowledge base a
- the user wants to get the value of an attribute for a specific object a
 a
The action is not able to compare objects or consider relations between objects in your knowledge base. a
Furthermore, resolving any mention to the last mentioned object in the conversation might not always be optimal. a
 a
If you want to tackle more complex use cases, you can write your own custom action. a
We added some helper functions to ``rasa_sdk.knowledge_base.utils`` a
(`link to code <https://github.com/RasaHQ/rasa-sdk/tree/master/rasa_sdk/knowledge_base/>`_ ) a
to help you when implement your own solution. a
We recommend using ``KnowledgeBase`` interface so that you can still use the ``ActionQueryKnowledgeBase`` a
alongside your new custom action. a
 a
If you write a knowledge base action that tackles one of the above use cases or a new one, be sure to tell us about a
it on the `forum <https://forum.rasa.com>`_! a
 a
 a
.. _customize_in_memory_knowledge_base: a
 a
Customizing the ``InMemoryKnowledgeBase`` a
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ a
 a
The class ``InMemoryKnowledgeBase`` inherits ``KnowledgeBase``. a
You can customize your ``InMemoryKnowledgeBase`` by overwriting the following functions: a
 a
- ``get_key_attribute_of_object()``: To keep track of what object the user was talking about last, we store the value a
  of the key attribute in a specific slot. Every object should have a key attribute that is unique, a
  similar to the primary key in a relational database. By default, the name of the key attribute for every object type a
  is set to ``id``. You can overwrite the name of the key attribute for a specific object type by calling a
  ``set_key_attribute_of_object()``. a
- ``get_representation_function_of_object()``: Let's focus on the following restaurant: a
 a
  .. code-block:: json a
 a
      { a
          "id": 0, a
          "name": "Donath", a
          "cuisine": "Italian", a
          "outside-seating": true, a
          "price-range": "mid-range" a
      } a
 a
  When the user asks the bot to list any Italian restaurant, it doesn't need all of the details of the restaurant. a
  Instead, you want to provide a meaningful name that identifies the restaurant -- in most cases, the name of the object will do. a
  The function ``get_representation_function_of_object()`` returns a lambda function that maps the a
  above restaurant object to its name. a
 a
  .. code-block:: python a
 a
      lambda obj: obj["name"] a
 a
  This function is used whenever the bot is talking about a specific object, so that the user is presented a meaningful a
  name for the object. a
 a
  By default, the lambda function returns the value of the ``"name"`` attribute of the object. a
  If your object does not have a ``"name"`` attribute , or the ``"name"`` of an object is a
  ambiguous, you should set a new lambda function for that object type by calling a
  ``set_representation_function_of_object()``. a
- ``set_ordinal_mention_mapping()``: The ordinal mention mapping is needed to resolve an ordinal mention, such as a
  "second one," to an object in a list. By default, the ordinal mention mapping looks like this: a
 a
  .. code-block:: python a
 a
      { a
          "1": lambda l: l[0], a
          "2": lambda l: l[1], a
          "3": lambda l: l[2], a
          "4": lambda l: l[3], a
          "5": lambda l: l[4], a
          "6": lambda l: l[5], a
          "7": lambda l: l[6], a
          "8": lambda l: l[7], a
          "9": lambda l: l[8], a
          "10": lambda l: l[9], a
          "ANY": lambda l: random.choice(l), a
          "LAST": lambda l: l[-1], a
      } a
 a
  You can overwrite it by calling the function ``set_ordinal_mention_mapping()``. a
  If you want to learn more about how this mapping is used, check out :ref:`resolve_mentions`. a
 a
 a
See the `example bot <https://github.com/RasaHQ/rasa/blob/master/examples/knowledgebasebot/actions.py>`_ for an a
example implementation of an ``InMemoryKnowledgeBase`` that uses the method ``set_representation_function_of_object()`` a
to overwrite the default representation of the object type "hotel." a
The implementation of the ``InMemoryKnowledgeBase`` itself can be found in the a
`rasa-sdk <https://github.com/RasaHQ/rasa-sdk/tree/master/rasa_sdk/knowledge_base/>`_ package. a
 a
 a
.. _custom_knowledge_base: a
 a
Creating Your Own Knowledge Base a
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ a
 a
If you have more data or if you want to use a more complex data structure that, for example, involves relations between a
different objects, you can create your own knowledge base implementation. a
Just inherit ``KnowledgeBase`` and implement the methods ``get_objects()``, ``get_object()``, and a
``get_attributes_of_object()``. The `knowledge base code <https://github.com/RasaHQ/rasa-sdk/tree/master/rasa_sdk/knowledge_base/>`_ a
provides more information on what those methods should do. a
 a
You can also customize your knowledge base further, by adapting the methods mentioned in the section a
:ref:`customize_in_memory_knowledge_base`. a
 a
.. note:: a
   We wrote a `blog post <https://blog.rasa.com/set-up-a-knowledge-base-to-encode-domain-knowledge-for-rasa/>`_ a
   that explains how you can set up your own knowledge base. a
 a