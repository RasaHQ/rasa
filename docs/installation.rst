.. _section_backends:

Installation
============

Rasa NLU itself doesn't have any external requirements,
but to do something useful with it you need to
install & configure a backend. Which backend you want to use is up to you.

Setting up rasa NLU
~~~~~~~~~~~~~~~~~~~
The recommended way to install rasa NLU is using pip:

.. code-block:: bash

    pip install rasa_nlu

If you want to use the bleeding edge version use github + setup.py:

.. code-block:: bash

    git clone https://github.com/RasaHQ/rasa_nlu.git
    cd rasa_nlu
    pip install -r requirements.txt
    pip install -e .

rasa NLU allows you to use components to process your messages.
E.g. there is a component for intent classification and
there are several different components for entity recognition.
The different components have their own requirements. To get
you started quickly, this installation guide only installs
the basic requirements, you may need to install other
dependencies if you want to use certain components. When running
rasa NLU it will check if all needed dependencies are
installed and tell you which are missing, if any.

.. note::

    If you want to make sure you got all the dependencies
    installed any component might ever need, and you
    don't mind the additional dependencies lying around, you can use

    .. code-block:: bash

        pip install -r alt_requirements/requirements_full.txt

    instead of ``requirements.txt`` to install all requirements.

Setting up a backend
~~~~~~~~~~~~~~~~~~~~
Most of the processing pipeline you can use with rasa NLU
either require MITIE, spaCy or sklearn to be installed.

Best for most: spaCy + sklearn
------------------------------

You can also run using these two in combination. 

installing spacy just requires (for more information
visit the `spacy docu <https://spacy.io/docs/usage/>`_):

.. code-block:: bash

    pip install rasa_nlu[spacy]
    python -m spacy download en_core_web_md
    python -m spacy link en_core_web_md en

We highly recommend using at least the "medium" sized models (``_md``)instead of
the spacy's default small ``en_core_web_sm`` model. Small models will
work as well, the downside is that
they have worse performance during intent classification.

.. note::

    Using spaCy as the backend for Rasa is the **preferred option**.
    For most domains the performance is better or equally
    good as results achieved with MITIE. Additionally,
    it is easier to setup and faster to train.

First Alternative: MITIE
-------------------------

The `MITIE <https://github.com/mit-nlp/MITIE>`_ backend is all-inclusive,
in the sense that it provides both the NLP and the ML parts.

.. code-block:: bash

    pip install git+https://github.com/mit-nlp/MITIE.git
    pip install rasa_nlu[mitie]


and then download the `MITIE models <https://github.com/mit-nlp/MITIE/releases/download/v0.4/MITIE-models-v0.2.tar.bz2>`_.
The file you need is ``total_word_feature_extractor.dat``. Save this
somewhere and in your ``config.json`` add
``'mitie_file' : '/path/to/total_word_feature_extractor.dat'``.

.. warning::

    Training MITIE can be quite slow on datasets
    with more than a few intents. You can try
        - to use the sklearn + MITIE backend instead
          (which uses sklearn for the training) or
        - you can install `our mitie fork <https://github.com/tmbo/mitie>`_
          which should reduce the training time as well.

Another Alternative: sklearn + MITIE
------------------------------------
There is a third backend that combines the advantages of the two previous ones:

1. the fast and good intent classification from sklearn and
2. the good entitiy recognition and feature vector creation from MITIE

Especially, if you have a larger number of intents (more than 10),
training intent classifiers with MITIE can take very long.

To use this backend you need to follow the instructions for
installing both, sklearn and MITIE.
