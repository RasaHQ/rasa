:desc: Use open source named entity recognition like Spacy or Duckling
       and customize them according to your needs to build contextual
       AI assistants

.. _entity-extraction:

Entity Extraction
=================

.. edit-link::

Entity extraction involves parsing user messages for required pieces of information. Rasa Open Source
provides entity extractors for custom entities as well as pre-trained ones like dates and locations.
Here is a summary of the available extractors and what they are used for:

=========================  =================  ========================  =================================
Component                  Requires           Model           	        Notes
=========================  =================  ========================  =================================
``CRFEntityExtractor``     sklearn-crfsuite   conditional random field  good for training custom entities
``SpacyEntityExtractor``   spaCy              averaged perceptron       provides pre-trained entities
``DucklingHTTPExtractor``  running duckling   context-free grammar      provides pre-trained entities
``MitieEntityExtractor``   MITIE              structured SVM            good for training custom entities
``EntitySynonymMapper``    existing entities  N/A                       maps known synonyms
``DIETClassifier``                            conditional random field
                                              on top of a transformer   good for training custom entities
=========================  =================  ========================  =================================

.. contents::
   :local:

The "entity" Object
^^^^^^^^^^^^^^^^^^^

After parsing, an entity is returned as a dictionary. There are two fields that show information
about how the pipeline impacted the entities returned: the ``extractor`` field
of an entity tells you which entity extractor found this particular entity, and
the ``processors`` field contains the name of components that altered this
specific entity.

The use of synonyms can cause the ``value`` field not match the ``text``
exactly. Instead it will return the trained synonym.

.. code-block:: json

    {
      "text": "show me chinese restaurants",
      "intent": "restaurant_search",
      "entities": [
        {
          "start": 8,
          "end": 15,
          "value": "chinese",
          "entity": "cuisine",
          "extractor": "CRFEntityExtractor",
          "confidence": 0.854,
          "processors": []
        }
      ]
    }

.. note::

    The ``confidence`` will be set by the ``CRFEntityExtractor`` and the ``DIETClassifier`` component. The
    ``DucklingHTTPExtractor`` will always return ``1``. The ``SpacyEntityExtractor`` extractor
    does not provide this information and returns ``null``.


Some extractors, like ``duckling``, may include additional information. For example:

.. code-block:: json

   {
     "additional_info":{
       "grain":"day",
       "type":"value",
       "value":"2018-06-21T00:00:00.000-07:00",
       "values":[
         {
           "grain":"day",
           "type":"value",
           "value":"2018-06-21T00:00:00.000-07:00"
         }
       ]
     },
     "confidence":1.0,
     "end":5,
     "entity":"time",
     "extractor":"DucklingHTTPExtractor",
     "start":0,
     "text":"today",
     "value":"2018-06-21T00:00:00.000-07:00"
   }


Custom Entities
^^^^^^^^^^^^^^^

Almost every chatbot and voice app will have some custom entities.
A restaurant assistant should understand ``chinese`` as a cuisine,
but to a language-learning assistant it would mean something very different.
The ``CRFEntityExtractor`` and the ``DIETClassifier`` component can learn custom entities in any language, given
some training data.
See :ref:`training-data-format` for details on how to include entities in your training data.


.. _entities-roles-groups:

Entities Roles and Groups
^^^^^^^^^^^^^^^^^^^^^^^^^

.. warning::
   This feature is experimental.
   We introduce experimental features to get feedback from our community, so we encourage you to try it out!
   However, the functionality might be changed or removed in the future.
   If you have feedback (positive or negative) please share it with us on the `forum <https://forum.rasa.com>`_.

Assigning custom entity labels to words, allow you to define certain concepts in the data.
For example, we can define what a `city` is:

.. code-block:: none

    I want to fly from [Berlin](city) to [San Francisco](city).

However, sometimes you want to specify entities even further.
Let's assume we want to build an assistant that should book a flight for us.
The assistant needs to know which of the two cities in the example above is the departure city and which is the
destination city.
``Berlin`` and ``San Francisco`` are still cities, but they play a different role in our example.
To distinguish between the different roles, you can assign a role label in addition to the entity label.

.. code-block:: none

    - I want to fly from [Berlin]{"entity": "city", "role": "departure"} to [San Francisco]{"entity": "city", "role": "destination"}.

You can also group different entities by specifying a group label next to the entity label.
The group label can, for example, be used to define different orders.
In the following example we use the group label to reference what toppings goes with which pizza and
what size which pizza has.

.. code-block:: none

    Give me a [small]{"entity": "size", "group": "1"} pizza with [mushrooms]{"entity": "topping", "group": "1"} and
    a [large]{"entity": "size", "group": "2"} [pepperoni]{"entity": "topping", "group": "2"}

See :ref:`training-data-format` for details on how to define entities with roles and groups in your training data.

The entity object returned by the extractor will include the detected role/group label.

.. code-block:: json

    {
      "text": "Book a flight from Berlin to SF",
      "intent": "book_flight",
      "entities": [
        {
          "start": 19,
          "end": 25,
          "value": "Berlin",
          "entity": "city",
          "role": "departure",
          "extractor": "DIETClassifier",
        },
        {
          "start": 29,
          "end": 31,
          "value": "San Francisco",
          "entity": "city",
          "role": "destination",
          "extractor": "DIETClassifier",
        }
      ]
    }

.. note::

    Composite entities are currently only supported by the :ref:`diet-classifier` and :ref:`CRFEntityExtractor`.

In order to properly train your model with entities that have roles/groups, make sure to include enough training data
examples for every combination of entity and role/group label.
Also make sure to have some variations in your training data, so that the model is able to generalize.
For example, you should not only have example like ``fly FROM x TO y``, but also include examples like
``fly TO y FROM x``.

To fill slots from entities with a specific role/group, you need to either define a custom slot mappings using
:ref:`forms` or use :ref:`custom-actions` to extract the corresponding entity directly from the tracker.


Extracting Places, Dates, People, Organizations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

spaCy has excellent pre-trained named-entity recognizers for a few different languages.
You can test them out in this
`interactive demo <https://explosion.ai/demos/displacy-ent>`_.
We don't recommend that you try to train your own NER using spaCy,
unless you have a lot of data and know what you are doing.
Note that some spaCy models are highly case-sensitive.

Dates, Amounts of Money, Durations, Distances, Ordinals
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The `duckling <https://duckling.wit.ai/>`_ library does a great job
of turning expressions like "next Thursday at 8pm" into actual datetime
objects that you can use, e.g.

.. code-block:: python

   "next Thursday at 8pm"
   => {"value":"2018-05-31T20:00:00.000+01:00"}


The list of supported languages can be found `here
<https://github.com/facebook/duckling/tree/master/Duckling/Dimensions>`_.
Duckling can also handle durations like "two hours",
amounts of money, distances, and ordinals.
Fortunately, there is a duckling docker container ready to use,
that you just need to spin up and connect to Rasa NLU
(see :ref:`DucklingHTTPExtractor`).


Regular Expressions (regex)
^^^^^^^^^^^^^^^^^^^^^^^^^^^

You can use regular expressions to help the CRF model learn to recognize entities.
In your training data (see :ref:`training-data-format`) you can provide a list of regular expressions, each of which provides
the ``CRFEntityExtractor`` with an extra binary feature, which says if the regex was found (1) or not (0).

For example, the names of German streets often end in ``strasse``. By adding this as a regex,
we are telling the model to pay attention to words ending this way, and will quickly learn to
associate that with a location entity.

If you just want to match regular expressions exactly, you can do this in your code,
as a postprocessing step after receiving the response from Rasa NLU.


.. _entity-extraction-custom-features:

Passing Custom Features to ``CRFEntityExtractor``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you want to pass custom features, such as pre-trained word embeddings, to ``CRFEntityExtractor``, you can
add any dense featurizer to the pipeline before the ``CRFEntityExtractor``.
``CRFEntityExtractor`` automatically finds the additional dense features and checks if the dense features are an
iterable of ``len(tokens)``, where each entry is a vector.
A warning will be shown in case the check fails.
However, ``CRFEntityExtractor`` will continue to train just without the additional custom features.
In case dense features are present, ``CRFEntityExtractor`` will pass the dense features to ``sklearn_crfsuite``
and use them for training.
