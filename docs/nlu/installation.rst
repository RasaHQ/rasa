:desc: Manage our open source NLU on premise to allow local intent recognition,
       entity extraction and customisation of the language models.
:meta_image: https://i.imgur.com/nGF1K8f.jpg

.. _section_backends:

Installation
============


Prerequisites
~~~~~~~~~~~~~
For windows
-----------

Make sure the Microsoft VC++ Compiler is installed, so python can compile
any dependencies. You can get the compiler from:
https://visualstudio.microsoft.com/visual-cpp-build-tools/
Download the installer and select VC++ Build tools in the list.

Setting up Rasa NLU
~~~~~~~~~~~~~~~~~~~

Stable (Recommended)
--------------------
The recommended way to install Rasa NLU is using pip which will install
the latest stable version of Rasa NLU:

.. copyable::

    pip install rasa

Latest (Most recent github)
---------------------------
If you want to use the bleeding edge version you can get it from github:

.. code-block:: bash

    git clone https://github.com/RasaHQ/rasa_nlu.git
    cd rasa_nlu
    pip install -r requirements.txt
    pip install -e .

Rasa NLU has different components for recognizing intents and entities,
most of these will have some additional dependencies.

When you train your model, Rasa NLU will check if all required dependencies are
installed and tell you if any are missing.

.. note::

    If you want to make sure you have the dependencies
    installed for any component you might ever need, and you
    don't mind the additional dependencies lying around, you can use

    .. code-block:: bash

        pip install -r alt_requirements/requirements_full.txt

    to install everything.


Installing Pipeline Dependencies
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Section :ref:`section_pipeline` will help you choose which pipeline
you want to use.

Great for getting started: pretrained embeddings from spaCy
-----------------------------------------------------------


The ``pretrained_embeddings_spacy`` pipeline combines a few different libraries and
is a popular option.

You can install it with this command (for more information
visit the `spacy docs <https://spacy.io/usage/models>`_):

.. code-block:: bash

    pip install rasa[spacy]
    python -m spacy download en_core_web_md
    python -m spacy link en_core_web_md en

This will install Rasa NLU as well as spacy and its language model
for the english language. We recommend using at least the
"medium" sized models (``_md``) instead of the spacy's
default small ``en_core_web_sm`` model. Small models require less
memory to run, but will somewhat reduce intent classification performance.



First Alternative: Tensorflow
-----------------------------

To use the ``supervised_embeddings`` pipeline you will need to install
Tensorflow and, for entity recognition, the sklearn-crfsuite library.
To do this, run the following command:

.. code-block:: bash

    pip install rasa


Second Alternative: MITIE
-------------------------

The `MITIE <https://github.com/mit-nlp/MITIE>`_ backend performs well for
small datasets, but training can take very long if you have more than a
couple of hundred examples. We may deprecate the MITIE backend in the future.

First, run

.. code-block:: bash

    pip install git+https://github.com/mit-nlp/MITIE.git
    pip install rasa[mitie]

and then download the
`MITIE models <https://github.com/mit-nlp/MITIE/releases/download/v0.4/MITIE-models-v0.2.tar.bz2>`_.
The file you need is ``total_word_feature_extractor.dat``. Save this
somewhere, if you want to use mitie, you need to tell it where to
find this file.

The complete pipeline for mitie can be found here

.. literalinclude:: ../../sample_configs/config_pretrained_embeddings_mitie.yml
    :language: yaml

Using MITIE alone can be quite slow to train, but you can use it with this configuration


.. literalinclude:: ../../sample_configs/config_pretrained_embeddings_mitie_2.yml
    :language: yaml



Train your first custom Rasa NLU model!
---------------------------------------

After following the quickstart and installing Rasa NLU, the next step is to 
build something yourself! To get you started, we have prepared a 
Rasa NLU starter-pack which has all the files you need to train your
first custom Rasa NLU model.
On top of that, the starter-pack includes a training dataset ready 
for you to use.

Click the link below to get the Rasa NLU starter-pack:
	
`Rasa NLU starter-pack <https://github.com/RasaHQ/starter-pack-rasa-nlu>`_
	
Let us know how you are getting on! If you have any questions
about the starter-pack or using Rasa NLU in general, post your questions on
the `Rasa Community Forum <https://forum.rasa.com>`_!

.. include:: feedback.inc
