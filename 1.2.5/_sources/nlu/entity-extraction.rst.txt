:desc: Use open source named entity recognition like Spacy or Duckling
       and customize them according to your needs to build contextual
       AI assistants

.. _entity-extraction:

Entity Extraction
=================

.. edit-link::

.. contents::
   :local:


Introduction
^^^^^^^^^^^^

Here is a summary of the available extractors and what they are used for:

=========================  =================  ========================  =================================
Component                  Requires           Model           	        Notes
=========================  =================  ========================  =================================
``CRFEntityExtractor``     sklearn-crfsuite   conditional random field  good for training custom entities
``SpacyEntityExtractor``   spaCy              averaged perceptron       provides pre-trained entities
``DucklingHTTPExtractor``  running duckling   context-free grammar      provides pre-trained entities
``MitieEntityExtractor``   MITIE              structured SVM            good for training custom entities
``EntitySynonymMapper``    existing entities  N/A                       maps known synonyms
=========================  =================  ========================  =================================

If your pipeline includes one or more of the components above,
the output of your trained model will include the extracted entities as well
as some metadata about which component extracted them.
The ``processors`` field contains the names of components that altered each entity.

.. note::
   The ``value`` field can be different from what appears in the text.
   If you use synonyms, an extracted entity like ``chinees`` will be mapped
   to a standard value, e.g. ``chinese``.

Here is an example response:

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


Custom Entities
^^^^^^^^^^^^^^^

Almost every chatbot and voice app will have some custom entities.
A restaurant assistant should understand ``chinese`` as a cuisine,
but to a language-learning assistant it would mean something very different.
The ``CRFEntityExtractor`` component can learn custom entities in any language, given
some training data.
See :ref:`training-data-format` for details on how to include entities in your training data.


Extracting Places, Dates, People, Organisations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

spaCy has excellent pre-trained named-entity recognisers for a few different languages.
You can test them out in this
`interactive demo <https://demos.explosion.ai/displacy-ent/>`_.
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


The list of supported langauges can be found `here 
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
