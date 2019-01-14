:desc: Installing Rasa NLU on Mac, Windows, and Linux
:meta_image: https://i.imgur.com/nGF1K8f.jpg

.. _section_backends:

Installation
============


Prerequisites
~~~~~~~~~~~~~
For windows
-----------
Make sure the Microsoft VC++ Compiler is installed, so python can compile any dependencies. You can get the compiler from: 
https://visualstudio.microsoft.com/visual-cpp-build-tools/
Download the installer and select VC++ Build tools in the list. 

Setting up Rasa NLU
~~~~~~~~~~~~~~~~~~~

Stable (Recommended)
--------------------
The recommended way to install Rasa NLU is using pip which will install the latest stable version of Rasa NLU:

.. copyable::

    pip install rasa_nlu

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

Section :ref:`section_pipeline` will help you choose which pipeline you want to use. 

Great for getting started: spaCy + sklearn
------------------------------------------


The ``spacy_sklearn`` pipeline combines a few different libraries and is a popular option.

You can install it with this command (for more information
visit the `spacy docs <https://spacy.io/usage/models>`_):

.. code-block:: bash

    pip install rasa_nlu[spacy]
    python -m spacy download en_core_web_md
    python -m spacy link en_core_web_md en

This will install Rasa NLU as well as spacy and its language model
for the english language. We recommend using at least the
"medium" sized models (``_md``) instead of the spacy's
default small ``en_core_web_sm`` model. Small models require less 
memory to run, but will somewhat reduce intent classification performance.



First Alternative: Tensorflow
-----------------------------

To use the ``tensorflow_embedding`` pipeline you will need to install tensorflow as well as the scikit-learn and sklearn-crfsuite libraries. To do this, run the following command:

.. code-block:: bash

    pip install rasa_nlu[tensorflow]


Second Alternative: MITIE
-------------------------

The `MITIE <https://github.com/mit-nlp/MITIE>`_ backend performs well for small datasets, but training can take very long if you have more than a couple of hundred examples. We may deprecate the MITIE backend in the future. 

First, run

.. code-block:: bash

    pip install git+https://github.com/mit-nlp/MITIE.git
    pip install rasa_nlu[mitie]

and then download the `MITIE models <https://github.com/mit-nlp/MITIE/releases/download/v0.4/MITIE-models-v0.2.tar.bz2>`_.
The file you need is ``total_word_feature_extractor.dat``. Save this
somewhere, if you want to use mitie, you need to tell it where to
find this file.

The complete pipeline for mitie can be found here

.. literalinclude:: ../sample_configs/config_mitie.yml
    :language: yaml

.. warning::

    Training MITIE can be quite slow on datasets
    with more than a few intents. You can try

        - to use the sklearn + MITIE backend instead
          (which uses sklearn for the training) or
        - you can install `our mitie fork <https://github.com/tmbo/mitie>`_
          which should reduce the training time as well.

Another Alternative: sklearn + MITIE
------------------------------------
There is another backend that combines the advantages of the two previous ones:

1. the fast and good intent classification from sklearn and
2. the good entitiy recognition and feature vector creation from MITIE

Especially, if you have a larger number of intents (more than 10),
training intent classifiers with MITIE can take very long.

To use this backend you need to follow the instructions for
installing both, sklearn and MITIE.

Example pipeline configuration for the use of MITIE together with
sklearn:

.. literalinclude:: ../sample_configs/config_mitie_sklearn.yml
    :language: yaml


Train your first custom Rasa NLU model!
---------------------------------------
After following the quickstart and installing Rasa NLU, the next step is to 
build something yourself! To get you started, we have prepared a 
Rasa NLU starter-pack which has all the files you need to train your first custom Rasa NLU model.
On top of that, the starter-pack includes a training dataset ready 
for you to use.

Click the link below to get the Rasa NLU starter-pack:
	
`Rasa NLU starter-pack <https://github.com/RasaHQ/starter-pack-rasa-nlu>`_
	
Let us know how you are getting on! If you have any questions about the starter-pack or 
using Rasa NLU in general, post your questions on `Rasa Community Forum <https://forum.rasa.com>`_!    


.. include:: feedback.inc


