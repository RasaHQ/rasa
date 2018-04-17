.. _section_backends:

Installation
============

rasa NLU itself doesn't have any external requirements, but to do something useful with it you need to install & configure a backend. 

Setting up rasa NLU
~~~~~~~~~~~~~~~~~~~
The recommended way to install rasa NLU is using pip:

.. code-block:: bash

    pip install rasa_nlu

If you want to use the bleeding edge version use github + setup.py:

.. code-block:: bash

    git clone git@github.com:golastmile/rasa_nlu.git
    cd rasa_nlu
    python setup.py install


Setting up a backend
~~~~~~~~~~~~~~~~~~~~

Option 1 : MITIE
----------------

The `MITIE <https://github.com/mit-nlp/MITIE>`_ backend is all-inclusive, in the sense that it provides both the NLP and the ML parts.

.. code-block:: bash

    pip install git+https://github.com/mit-nlp/MITIE.git


and then download the `MITIE models <https://github.com/mit-nlp/MITIE/releases/download/v0.4/MITIE-models-v0.2.tar.bz2>`_ . 
The file you need is ``total_word_feature_extractor.dat``. Save this somewhere and in your ``config.json`` add ``'mitie_file' : '/path/to/total_word_feature_extractor.dat'``.

Option 2 : spaCy + sklearn
--------------------------

You can also run using these two in combination. 

installing spacy just requires:

.. code-block:: bash

    pip install -U spacy
    python -m spacy.en.download all

If you haven't used ``numpy/scipy`` before, it is highly recommended that you use conda.
steps are

- install `anaconda <https://www.continuum.io/downloads>`_
- ``conda install scikit-learn``

otherwise if you know what you're doing, you can also just ``pip install -U scikit-learn``

Option 3 : sklearn + MITIE
----------------------------------
There is a third backend that combines the advantages of the two previous ones:

1. the fast and good intent classification from sklearn and
2. the good entitiy recognition and feature vector creation from MITIE

Especially, if you have a larger number of intents (more than 10), training intent classifiers with MITIE can take very
long.

To use this backend you need to follow the instructions for installing both, sklearn and MITIE.
