.. _section_languages:

Language Support
================
Currently Rasa NLU is tested and readily available for the following languages:

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
allow Rasa NLU to load and use your new language by passing in your language identifier as the ``language`` :ref:`section_configuration` option.
