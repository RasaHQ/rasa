.. _section_showcase:

Important Features
==================

Domain-Specific Word Vectors
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Using pre-trained word embeddings to classify text has excellent performance when you have 
only a little bit of training data. But this has some drawbacks as well. 
The ``tensorflow_embedding`` pipeline can learn word vectors for words that are specific to your domain,
so that a banking chatbot can understand that ``"balance"`` is more closely related to ``"account"`` than to ``"symmetry"``. 
See `this blog post <https://medium.com/rasa-blog/supervised-word-vectors-from-scratch-in-rasa-nlu-6daf794efcd8>`_ for details. 

Extracting Multiple Intents from one Message
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Sometimes your users will say more than one thing at a time. For example, *"thanks! any other suggestions"* expresses both a ``thankyou`` and a ``ask_more_suggestions`` intent. Rasa can predict multiple labels for each message! 
See `this blog post <https://medium.com/rasa-blog/supervised-word-vectors-from-scratch-in-rasa-nlu-6daf794efcd8>`_ for details. 

Custom Entities
^^^^^^^^^^^^^^^

Almost every chatbot and voice app will have some custom entities.
In a restaurant bot, ``chinese`` is a cuisine, but in a language-learning app it would mean something very different. 
The ``ner_crf`` component can learn custom entities in any language. 

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

The `duckling <https://duckling.wit.ai/>`_ package does a great job
of turning expressions like "next Thursday at 8pm" into actual datetime
objects that you can use. The list of supported langauges is here: TODO
Duckling can also handle durations like "two hours",
amounts of money, distances, etc. Fortunately, there is also a
`python wrapper <https://github.com/FraBle/python-duckling>`_ for
duckling! You can use this component by installing the duckling
package from PyPI and adding ``ner_duckling`` to your pipeline.


