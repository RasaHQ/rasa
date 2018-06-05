.. _section_entities:

Entity Extraction
=================


Custom Entities
^^^^^^^^^^^^^^^

Almost every chatbot and voice app will have some custom entities.
In a restaurant bot, ``chinese`` is a cuisine, but in a language-learning app it would mean something very different. 
The ``ner_crf`` component can learn custom entities in any language. 

Regular Expressions (regex)
^^^^^^^^^^^^^^^^^^^^^^^^^^^



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
Fortunately, there is also a
`python wrapper <https://github.com/FraBle/python-duckling>`_ for
duckling! You can use this component by installing the duckling
package from PyPI and adding ``ner_duckling`` to your pipeline.
Alternatively, you can run duckling separately (natively or in a docker container)
and use the ``ner_duckling_http`` component. 

