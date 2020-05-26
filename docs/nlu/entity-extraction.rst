:desc: Use open source named entity recognition like Spacy or Duckling a
       and customize them according to your needs to build contextual a
       AI assistants a
 a
.. _entity-extraction: a
 a
Entity Extraction a
================= a
 a
.. edit-link:: a
 a
Entity extraction involves parsing user messages for required pieces of information. Rasa Open Source a
provides entity extractors for custom entities as well as pre-trained ones like dates and locations. a
Here is a summary of the available extractors and what they are used for: a
 a
=========================  =================  ========================  ================================= a
Component                  Requires           Model           	        Notes a
=========================  =================  ========================  ================================= a
``CRFEntityExtractor``     sklearn-crfsuite   conditional random field  good for training custom entities a
``SpacyEntityExtractor``   spaCy              averaged perceptron       provides pre-trained entities a
``DucklingHTTPExtractor``  running duckling   context-free grammar      provides pre-trained entities a
``MitieEntityExtractor``   MITIE              structured SVM            good for training custom entities a
``EntitySynonymMapper``    existing entities  N/A                       maps known synonyms a
``DIETClassifier``                            conditional random field a
                                              on top of a transformer   good for training custom entities a
=========================  =================  ========================  ================================= a
 a
.. contents:: a
   :local: a
 a
The "entity" Object a
^^^^^^^^^^^^^^^^^^^ a
 a
After parsing, an entity is returned as a dictionary. There are two fields that show information a
about how the pipeline impacted the entities returned: the ``extractor`` field a
of an entity tells you which entity extractor found this particular entity, and a
the ``processors`` field contains the name of components that altered this a
specific entity. a
 a
The use of synonyms can cause the ``value`` field not match the ``text`` a
exactly. Instead it will return the trained synonym. a
 a
.. code-block:: json a
 a
    { a
      "text": "show me chinese restaurants", a
      "intent": "restaurant_search", a
      "entities": [ a
        { a
          "start": 8, a
          "end": 15, a
          "value": "chinese", a
          "entity": "cuisine", a
          "extractor": "CRFEntityExtractor", a
          "confidence": 0.854, a
          "processors": [] a
        } a
      ] a
    } a
 a
.. note:: a
 a
    The ``confidence`` will be set by the ``CRFEntityExtractor`` component. The a
    ``DucklingHTTPExtractor`` will always return ``1``. The ``SpacyEntityExtractor`` extractor a
    and ``DIETClassifier`` do not provide this information and return ``null``. a
 a
 a
Some extractors, like ``duckling``, may include additional information. For example: a
 a
.. code-block:: json a
 a
   { a
     "additional_info":{ a
       "grain":"day", a
       "type":"value", a
       "value":"2018-06-21T00:00:00.000-07:00", a
       "values":[ a
         { a
           "grain":"day", a
           "type":"value", a
           "value":"2018-06-21T00:00:00.000-07:00" a
         } a
       ] a
     }, a
     "confidence":1.0, a
     "end":5, a
     "entity":"time", a
     "extractor":"DucklingHTTPExtractor", a
     "start":0, a
     "text":"today", a
     "value":"2018-06-21T00:00:00.000-07:00" a
   } a
 a
 a
Custom Entities a
^^^^^^^^^^^^^^^ a
 a
Almost every chatbot and voice app will have some custom entities. a
A restaurant assistant should understand ``chinese`` as a cuisine, a
but to a language-learning assistant it would mean something very different. a
The ``CRFEntityExtractor`` and the ``DIETClassifier`` component can learn custom entities in any language, given a
some training data. a
See :ref:`training-data-format` for details on how to include entities in your training data. a
 a
 a
.. _entities-roles-groups: a
 a
Entities Roles and Groups a
^^^^^^^^^^^^^^^^^^^^^^^^^ a
 a
.. warning:: a
   This feature is experimental. a
   We introduce experimental features to get feedback from our community, so we encourage you to try it out! a
   However, the functionality might be changed or removed in the future. a
   If you have feedback (positive or negative) please share it with us on the `forum <https://forum.rasa.com>`_. a
 a
Assigning custom entity labels to words, allow you to define certain concepts in the data. a
For example, we can define what a `city` is: a
 a
.. code-block:: none a
 a
    I want to fly from [Berlin](city) to [San Francisco](city). a
 a
However, sometimes you want to specify entities even further. a
Let's assume we want to build an assistant that should book a flight for us. a
The assistant needs to know which of the two cities in the example above is the departure city and which is the a
destination city. a
``Berlin`` and ``San Francisco`` are still cities, but they play a different role in our example. a
To distinguish between the different roles, you can assign a role label in addition to the entity label. a
 a
.. code-block:: none a
 a
    - I want to fly from [Berlin]{"entity": "city", "role": "departure"} to [San Francisco]{"entity": "city", "role": "destination"}. a
 a
You can also group different entities by specifying a group label next to the entity label. a
The group label can, for example, be used to define different orders. a
In the following example we use the group label to reference what toppings goes with which pizza and a
what size which pizza has. a
 a
.. code-block:: none a
 a
    Give me a [small]{"entity": "size", "group": "1"} pizza with [mushrooms]{"entity": "topping", "group": "1"} and a
    a [large]{"entity": "size", "group": "2"} [pepperoni]{"entity": "topping", "group": "2"} a
 a
See :ref:`training-data-format` for details on how to define entities with roles and groups in your training data. a
 a
The entity object returned by the extractor will include the detected role/group label. a
 a
.. code-block:: json a
 a
    { a
      "text": "Book a flight from Berlin to SF", a
      "intent": "book_flight", a
      "entities": [ a
        { a
          "start": 19, a
          "end": 25, a
          "value": "Berlin", a
          "entity": "city", a
          "role": "departure", a
          "extractor": "DIETClassifier", a
        }, a
        { a
          "start": 29, a
          "end": 31, a
          "value": "San Francisco", a
          "entity": "city", a
          "role": "destination", a
          "extractor": "DIETClassifier", a
        } a
      ] a
    } a
 a
.. note:: a
 a
    Composite entities are currently only supported by the :ref:`diet-classifier` and :ref:`CRFEntityExtractor`. a
 a
In order to properly train your model with entities that have roles/groups, make sure to include enough training data a
examples for every combination of entity and role/group label. a
Also make sure to have some variations in your training data, so that the model is able to generalize. a
For example, you should not only have example like ``fly FROM x TO y``, but also include examples like a
``fly TO y FROM x``. a
 a
To fill slots from entities with a specific role/group, you need to either define a custom slot mappings using a
:ref:`forms` or use :ref:`custom-actions` to extract the corresponding entity directly from the tracker. a
 a
 a
Extracting Places, Dates, People, Organisations a
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ a
 a
spaCy has excellent pre-trained named-entity recognisers for a few different languages. a
You can test them out in this a
`interactive demo <https://explosion.ai/demos/displacy-ent>`_. a
We don't recommend that you try to train your own NER using spaCy, a
unless you have a lot of data and know what you are doing. a
Note that some spaCy models are highly case-sensitive. a
 a
Dates, Amounts of Money, Durations, Distances, Ordinals a
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ a
 a
The `duckling <https://duckling.wit.ai/>`_ library does a great job a
of turning expressions like "next Thursday at 8pm" into actual datetime a
objects that you can use, e.g. a
 a
.. code-block:: python a
 a
   "next Thursday at 8pm" a
   => {"value":"2018-05-31T20:00:00.000+01:00"} a
 a
 a
The list of supported languages can be found `here a
<https://github.com/facebook/duckling/tree/master/Duckling/Dimensions>`_. a
Duckling can also handle durations like "two hours", a
amounts of money, distances, and ordinals. a
Fortunately, there is a duckling docker container ready to use, a
that you just need to spin up and connect to Rasa NLU a
(see :ref:`DucklingHTTPExtractor`). a
 a
 a
Regular Expressions (regex) a
^^^^^^^^^^^^^^^^^^^^^^^^^^^ a
 a
You can use regular expressions to help the CRF model learn to recognize entities. a
In your training data (see :ref:`training-data-format`) you can provide a list of regular expressions, each of which provides a
the ``CRFEntityExtractor`` with an extra binary feature, which says if the regex was found (1) or not (0). a
 a
For example, the names of German streets often end in ``strasse``. By adding this as a regex, a
we are telling the model to pay attention to words ending this way, and will quickly learn to a
associate that with a location entity. a
 a
If you just want to match regular expressions exactly, you can do this in your code, a
as a postprocessing step after receiving the response from Rasa NLU. a
 a
 a
.. _entity-extraction-custom-features: a
 a
Passing Custom Features to ``CRFEntityExtractor`` a
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ a
 a
If you want to pass custom features, such as pre-trained word embeddings, to ``CRFEntityExtractor``, you can a
add any dense featurizer to the pipeline before the ``CRFEntityExtractor``. a
``CRFEntityExtractor`` automatically finds the additional dense features and checks if the dense features are an a
iterable of ``len(tokens)``, where each entry is a vector. a
A warning will be shown in case the check fails. a
However, ``CRFEntityExtractor`` will continue to train just without the additional custom features. a
In case dense features are present, ``CRFEntityExtractor`` will pass the dense features to ``sklearn_crfsuite`` a
and use them for training. a
 a