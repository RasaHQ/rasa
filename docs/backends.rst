.. _section_backends:

Setting up a backend
====================================

rasa_nlu itself doesn't have any external requirements, but to do something useful with it you need to install & configure a backend. 

Option 1 : MITIE
----------------------------

The `MITIE <https://github.com/mit-nlp/MITIE>`_ backend is all-inclusive, in the sense that it provides both the NLP and the ML parts.

.. code-block:: console

    pip install git+https://github.com/mit-nlp/MITIE.git


and then download the `MITIE models <https://github.com/mit-nlp/MITIE/releases/download/v0.4/MITIE-models-v0.2.tar.bz2>`_ . 
The file you need is `total_word_feature_extractor.dat`

Option 2 : spaCy + scikit-learn
-------------------------------------

You can also run using these two in combination. 
`spaCy <https://spacy.io/>`_ is an excellent library for NLP tasks.
`scikit-learn <http://scikit-learn.org>`_ is a popular ML library.

.. code-block:: console

    pip install -U spacy
    python -m spacy.en.download all
    pip install -U scikit-learn


If you haven't used ``numpy/scipy`` before, it is highly recommended to use conda.
steps are

- install `anaconda <https://www.continuum.io/downloads>`_
- ``conda install scikit-learn``
