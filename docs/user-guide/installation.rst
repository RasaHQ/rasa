:desc: Manage our open source NLU on premise to allow local intent recognition,
       entity extraction and customisation of the language models.
:meta_image: https://i.imgur.com/nGF1K8f.jpg

.. _installation:

Installation
============

.. edit-link::

Quick Installation
~~~~~~~~~~~~~~~~~~

You can get started using ``pip`` with the following command.


.. code-block:: bash

    $ pip install rasa-x --extra-index-url https://pypi.rasa.com/simple


| This will install both Rasa and Rasa X on your system.
| Once you're done with this, you can head over to the tutorial!

For a more detailed guide on setting up, scroll down to the section below.

.. button::
   :text: Next Step: Tutorial
   :link: ../rasa-tutorial/


Installation Guide with pip
~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. Install the Python development environment
#############################################

Check if your Python environment is already configured:

.. code-block:: bash

    $ python3 --version
    $ pip3 --version
    $ virtualenv --version

If these packages are already installed, skip to the next step.

.. tabs::

    .. tab:: Ubuntu

        .. code-block:: bash

            $ sudo apt update
            $ sudo apt install python3-dev python3-pip
            $ sudo pip3 install -U virtualenv

    .. tab:: mac OS

        Install the Homebrew package manager if you haven't already.

        .. code-block:: bash

            $ /usr/bin/ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"
            $ export PATH="/usr/local/bin:/usr/local/sbin:$PATH"
            $ brew update
            $ brew install python  # Python 3
            $ sudo pip3 install -U virtualenv  # system-wide install

    .. tab:: Windows

        .. raw:: html

            Make sure the Microsoft VC++ Compiler is installed, so python can compile
            any dependencies. You can get the compiler from <a class="reference external"
            href="https://visualstudio.microsoft.com/visual-cpp-build-tools/"
            target="_blank">Visual Studio</a>. Download the installer and select
            VC++ Build tools in the list.

        .. code-block:: bat

            C:\> pip3 install -U pip virtualenv


2. Create a virtual environment (strongly recommended)
######################################################

Tools like `virtualenv <https://virtualenv.pypa.io/en/latest/>`_ and `virtualenvwrapper <https://virtualenvwrapper.readthedocs.io/en/latest/>`_ provide isolated Python environments, which are more practical than installing packages systemwide. They also let you install packages without root privileges.

.. tabs::

    .. tab:: Ubuntu / mac OS

        Create a new virtual environment by choosing a Python interpreter and making a ``./venv`` directory to hold it:

        .. code-block:: bash

            $ virtualenv --system-site-packages -p python3 ./venv

        Activate the virtual environment:

        .. code-block:: bash

            $ source ./venv/bin/activate

    .. tab:: Windows

        Create a new virtual environment by choosing a Python interpreter and making a ``.\venv`` directory to hold it:

        .. code-block:: bat

            C:\> virtualenv --system-site-packages -p python3 ./venv

        Activate the virtual environment:

        .. code-block:: bat

            C:\> .\venv\Scripts\activate


3. Install Rasa and Rasa X
##########################

.. tabs::

    .. tab:: Inside a virtualenv

        To install both Rasa and Rasa X in one go:

        .. code-block:: bash

            (venv) $ pip install rasa-x --extra-index-url https://pypi.rasa.com/simple

        If you just want to install Rasa without Rasa X:

        .. code-block:: bash

            (venv) $ pip install rasa

    .. tab:: System-wide install

        To install both Rasa and Rasa X in one go:

        .. code-block:: bash

            $ pip3 install --user rasa-x --extra-index-url https://pypi.rasa.com/simple

        If you just want to install Rasa without Rasa X:

        .. code-block:: bash

            $ pip3 install --user rasa

.. note::

    That's it! Rasa is now installed.


Building from Source
~~~~~~~~~~~~~~~~~~~~

If you want to use the development version of Rasa, you can get it from GitHub:

.. code-block:: bash

    $ git clone https://github.com/RasaHQ/rasa.git
    $ cd rasa
    $ pip install -r requirements.txt
    $ pip install -e .


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
