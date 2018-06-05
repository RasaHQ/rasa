.. _section_languages:

Language Support
================

Rasa NLU can be used to understand any language, but some backends are 
restricted to specific languages.

The ``tensorflow_embedding`` pipeline can be used for any language, because
it trains custom word embeddings for your domain.

Pre-trained Word Vectors
------------------------

With the spaCy backend you can now load fastText vectors, which are available 
for `hundreds of languages <https://github.com/facebookresearch/fastText/blob/master/pretrained-vectors.md>`_.


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
Jieba-MITIE    chinese (``zh``) :ref:`* <jieba>`
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

MITIE
^^^^^

1. Get a ~clean language corpus (a Wikipedia dump works) as a set of text files
2. Build and run `MITIE Wordrep Tool`_ on your corpus. This can take several hours/days depending on your dataset and your workstation. You'll need something like 128GB of RAM for wordrep to run - yes that's alot: try to extend your swap.
3. Set the path of your new ``total_word_feature_extractor.dat`` as value of the *mitie_file* parameter in ``config_mitie.json``

.. _jieba:

Jieba-MITIE
^^^^^^^^^^^

Some notes about using the Jieba tokenizer together with MITIE on chinese
language data: To use it, you need a proper MITIE feature extractor, e.g.
``data/total_word_feature_extractor_zh.dat``. It should be trained
from a Chinese corpus using the MITIE wordrep tools
(takes 2-3 days for training).

For training, please build the
`MITIE Wordrep Tool`_.
Note that Chinese corpus should be tokenized first before feeding
into the tool for training. Close-domain corpus that best matches
user case works best.

A detailed instruction on how to train the model yourself can be found in
A trained model from Chinese Wikipedia Dump and Baidu Baike can be `crownpku <https://github.com/crownpku>`_  's
`blogpost <http://www.crownpku.com/2017/07/27/%E7%94%A8Rasa_NLU%E6%9E%84%E5%BB%BA%E8%87%AA%E5%B7%B1%E7%9A%84%E4%B8%AD%E6%96%87NLU%E7%B3%BB%E7%BB%9F.html>`_.

.. _`MITIE Wordrep Tool`: https://github.com/mit-nlp/MITIE/tree/master/tools/wordrep
