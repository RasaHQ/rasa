:desc: Use open source named entity recognition like spacy and duckling
       for building contextual AI Assistants.

.. _section_entities:

Entity Extraction
=================


=========================  ================  ========================  =================================
Component                  Requires          Model           	       Notes
=========================  ================  ========================  =================================
``CRFEntityExtractor``     sklearn-crfsuite  conditional random field  good for training custom entities
``SpacyEntityExtractor``   spaCy             averaged perceptron       provides pre-trained entities
``DucklingHTTPExtractor``  running duckling  context-free grammar      provides pre-trained entities
``MitieEntityExtractor``   MITIE             structured SVM            good for training custom entities
=========================  ================  ========================  =================================


Custom Entities
^^^^^^^^^^^^^^^

Almost every chatbot and voice app will have some custom entities.
In a restaurant bot, ``chinese`` is a cuisine, but in a language-learning app it would mean something very different.
The ``CRFEntityExtractor`` component can learn custom entities in any language.


Extracting Places, Dates, People, Organisations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

spaCy has excellent pre-trained named-entity recognisers for a few different langauges.
You can test them out in this
`awesome interactive demo <https://demos.explosion.ai/displacy-ent/>`_.
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


The list of supported langauges is `here <https://github.com/facebook/duckling/tree/master/Duckling/Dimensions>`_.
Duckling can also handle durations like "two hours",
amounts of money, distances, and ordinals.
Fortunately, there is a duckling docker container ready to use,
that you just need to spin up and connect to Rasa NLU.
(see :ref:`DucklingHTTPExtractor`)


Regular Expressions (regex)
^^^^^^^^^^^^^^^^^^^^^^^^^^^

You can use regular expressions to help the CRF model learn to recognize entities.
In the :ref:`section_dataformat` you can provide a list of regular expressions, each of which provides
the ``CRFEntityExtractor`` with an extra binary feature, which says if the regex was found (1) or not (0).

For example, the names of German streets often end in ``strasse``. By adding this as a regex,
we are telling the model to pay attention to words ending this way, and will quickly learn to
associate that with a location entity.

If you just want to match regular expressions exactly, you can do this in your code,
as a postprocessing step after receiving the response form Rasa NLU.


Returned Entities Object
------------------------
In the object returned after parsing there are two fields that show information
about how the pipeline impacted the entities returned. The ``extractor`` field
of an entity tells you which entity extractor found this particular entity.
The ``processors`` field contains the name of components that altered this
specific entity.

The use of synonyms can also cause the ``value`` field not match the ``text``
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

.. note::

    The `confidence` will be set by the CRF entity extractor
    (`CRFEntityExtractor` component). The duckling entity extractor will always return
    `1`. The `SpacyEntityExtractor` extractor does not provide this information and
    returns `null`.


.. include:: feedback.inc
