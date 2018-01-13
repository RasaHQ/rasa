.. _section_entities:

Entity Extraction
=================
There are a number of different entity extraction components, which can seem intimidating for new users.
Here we'll go through a few use cases and make recommendations of what to use. 

================    ==========  ========================    ===================================
Component           Requires    Model           	          notes
================    ==========  ========================    ===================================
``ner_mitie``       MITIE       structured SVM              good for training custom entities
``ner_crf``         crfsuite    conditional random field    good for training custom entities
``ner_spacy``       spaCy       averaged perceptron         provides pre-trained entities
``ner_duckling``    duckling    context-free grammar        provides pre-trained entities
================    ==========  ========================    ===================================

The exact required packages can be found in ``dev-requirements.txt`` and they should also be shown when they are missing
and a component is used that requires them.

To improve entity extraction, you can use regex features if your entities have a distinctive format (e.g. zipcodes).
More information can be found in the :ref:`section_dataformat`.

.. note::
    To use these components, you will probably want to define a custom pipeline, see :ref:`section_pipeline`.
    You can add multiple ner components to your pipeline; the results from each will be combined in the final output.

Use Cases
---------

Here we'll outline some common use cases for entity extraction, and make recommendations on which components to use.


Places, Dates, People, Organisations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

spaCy has excellent pre-trained named-entity recognisers in a number of models. You can test them out in this `awesome interactive demo <https://demos.explosion.ai/displacy-ent/>`_. We don't recommend that you try to train your own NER using spaCy, unless you have a lot of data and know what you are doing. Note that some spaCy models are highly case-sensitive.

Dates, Amounts of Money, Durations, Distances, Ordinals
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The `duckling <https://duckling.wit.ai/>`_ package does a great job of turning expressions like "next Thursday at 8pm" into actual datetime objects that you can use. It can also handle durations like "two hours", amounts of money, distances, etc. Fortunately, there is also a `python wrapper <https://github.com/FraBle/python-duckling>`_ for duckling! You can use this component by installing the duckling package from PyPI and adding ``ner_duckling`` to your pipeline.


Custom, Domain-specific entities
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In the introductory tutorial we build a restaurant bot, and create custom entities for location and cuisine.
The best components for training these domain-specific entity recognisers are the ``ner_mitie`` and ``ner_crf`` components. 
It is recommended that you experiment with both of these to see what works best for your data set. 

Returned Entities Object
------------------------
In the object returned after parsing there are two fields that show information about how the pipeline impacted the entities returned. The ``extractor`` field of an entity tells you which entity extractor found this particular entity. The ``processors`` field contains the name of components that altered this specific entity.

The use of synonyms can also cause the ``value`` field not match the ``text`` exactly. Instead it will return the trained synonym.

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
          "extractor": "ner_mitie",
          "processors": []
        }
      ]
    }
