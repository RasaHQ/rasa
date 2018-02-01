.. _section_languages:

Language Support
================
Currently rasa NLU is tested and readily available for the following languages:

=============  ==============================
backend        supported languages
=============  ==============================
spacy-sklearn  english (``en``),
               german (``de``),
               spanish (``es``),
               portuguese (``pt``),
               italian (``it``),
               dutch (``nl``),
               french (``fr``)
MITIE          english (``en``)
=============  ==============================

These languages can be set as part of the :ref:`section_configuration`.

Adding a new language
---------------------
We want to make the process of adding new languages as simple as possible to increase the number of
supported languages. Nevertheless, to use a language you either need a trained word representation or
you need to train that presentation on your own using a large corpus of text data in that language.

These are the steps necessary to add a new language:

spacy-sklearn
^^^^^^^^^^^^^

spaCy already provides a really good documentation page about `Adding languages <https://spacy.io/docs/usage/adding-languages>`_.
This will help you train a tokenizer and vocabulary for a new language in spaCy.

As described in the documentation, you need to register your language using ``set_lang_class()`` which will
allow rasa NLU to load and use your new language by passing in your language identifier as the ``language`` :ref:`section_configuration` option.

MITIE
^^^^^

1. Get a ~clean language corpus (a Wikipedia dump works) as a set of text files
2. Build and run `MITIE wordrep tool <https://github.com/mit-nlp/MITIE>`_ on your corpus. This can take several hours/days depending on your dataset and your workstation. You'll need something like 128GB of RAM for wordrep to run - yes that's alot: try to extend your swap.
3. Set the path of your new ``total_word_feature_extractor.dat`` as value of the *mitie_file* parameter in ``config_mitie.json``


