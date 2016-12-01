.. _section_backends:

Setting up a backend
====================================

rasa NLU itself doesn't have any external requirements, but to do something useful with it you need to install & configure a backend. 

Option 1 : MITIE
----------------------------

The `MITIE <https://github.com/mit-nlp/MITIE>`_ backend is all-inclusive, in the sense that it provides both the NLP and the ML parts.

.. code-block:: console

    pip install git+https://github.com/mit-nlp/MITIE.git


and then download the `MITIE models <https://github.com/mit-nlp/MITIE/releases/download/v0.4/MITIE-models-v0.2.tar.bz2>`_ . 
The file you need is ``total_word_feature_extractor.dat``. Save this somewhere and in your ``config.json`` add ``'mitie_file' : '/path/to/total_word_feature_extractor.dat'``.

Option 2 : spaCy + scikit-learn
-------------------------------------

You can also run using these two in combination. 

installing spacy just requires:

.. code-block:: console

    pip install -U spacy
    python -m spacy.en.download all

If you haven't used ``numpy/scipy`` before, it is highly recommended that you use conda.
steps are

- install `anaconda <https://www.continuum.io/downloads>`_
- ``conda install scikit-learn``

otherwise if you know what you're doing, you can also just ``pip install -U scikit-learn``
