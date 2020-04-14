:desc: Manage our open source NLU on premise to allow local intent recognition,
       entity extraction and customisation of the language models.
:meta_image: https://i.imgur.com/nGF1K8f.jpg

.. _installation:

Installation
============

The recommended way to get started with Rasa is via ``pip``:

.. copyable::

    pip install rasa-x --extra-index-url https://pypi.rasa.com/simple

This will install both Rasa and Rasa X.
If you don't want to use Rasa X, run ``pip install rasa`` instead.

.. raw:: html

     Unless you've already got numpy & scipy installed, we highly recommend
     that you install and use
     <a class="reference external" href="https://www.anaconda.com/download/"
     target="_blank">Anaconda</a>.

If you want to use the development version of Rasa, you can get it from GitHub:

.. code-block:: bash

    git clone https://github.com/RasaHQ/rasa.git
    cd rasa
    pip install -r requirements.txt
    pip install -e .

Windows Prerequisites
~~~~~~~~~~~~~~~~~~~~~

.. raw:: html

     Make sure the Microsoft VC++ Compiler is installed, so python can compile
     any dependencies. You can get the compiler from <a class="reference external"
     href="https://visualstudio.microsoft.com/visual-cpp-build-tools/"
     target="_blank">Visual Studio</a>. Download the installer and select
     VC++ Build tools in the list.


NLU Pipeline Dependencies
~~~~~~~~~~~~~~~~~~~~~~~~~

Rasa NLU has different components for recognizing intents and entities,
most of which have some additional dependencies.

When you train your NLU model, Rasa will check if all required dependencies are
installed and tell you if any are missing. The page on :ref:`choosing-a-pipeline`
will help you pick which pipeline to use.

.. note::

    If you want to make sure you have the dependencies
    installed for any component you might ever need, and you
    don't mind the additional dependencies lying around, you can use

    .. code-block:: bash

        pip install -r alt_requirements/requirements_full.txt

    to install everything.


Great for getting started: pretrained embeddings from spaCy
-----------------------------------------------------------


The ``pretrained_embeddings_spacy`` pipeline combines a few different libraries and
is a popular option. For more information
check out the `spaCy docs <https://spacy.io/usage/models>`_.

You can install it with the following commands:

.. code-block:: bash

    pip install rasa[spacy]
    python -m spacy download en_core_web_md
    python -m spacy link en_core_web_md en

This will install Rasa NLU as well as spacy and its language model
for the English language. We recommend using at least the
"medium" sized models (``_md``) instead of the spacy's
default small ``en_core_web_sm`` model. Small models require less
memory to run, but will somewhat reduce intent classification performance.



First Alternative: Tensorflow
-----------------------------

To use the ``supervised_embeddings`` pipeline you will need to install
Tensorflow and, for entity recognition, the sklearn-crfsuite library.
To do this, simply run the following command:

.. code-block:: bash

    pip install rasa


.. _install-mitie:

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
anywhere. If you want to use MITIE, you need to
tell it where to find this file (in this example it was saved in the
``data`` folder of the project directory).

The complete pipeline for MITIE can be found here

.. literalinclude:: ../../sample_configs/config_pretrained_embeddings_mitie.yml
    :language: yaml

Using MITIE alone can be quite slow to train, but you can use it with this configuration.


.. literalinclude:: ../../sample_configs/config_pretrained_embeddings_mitie_2.yml
    :language: yaml



Next Step
---------

Now that you have everything installed, head over to the tutorial!

.. button::
   :text: Next Step: Tutorial
   :link: ../rasa-tutorial/
