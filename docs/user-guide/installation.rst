:desc: Manage our open source NLU on premise to allow local intent recognition,
       entity extraction and customisation of the language models.
:meta_image: https://i.imgur.com/nGF1K8f.jpg

.. _installation:

============
Installation
============

.. edit-link::

Quick Installation
~~~~~~~~~~~~~~~~~~

You can install both Rasa and Rasa X using pip (requires Python 3.5.4 or higher).

.. code-block:: bash

    $ pip install rasa-x --extra-index-url https://pypi.rasa.com/simple

- Having trouble installing? Read our :ref:`step-by-step installation guide <installation_guide>`.
- You can also :ref:`build Rasa from source <build_from_source>`.
- For advanced installation options such as building from source and installation instructions for
  custom pipelines, head over :ref:`here <pipeline_dependencies>`.


When you're done installing, you can head over to the tutorial!

.. button::
   :text: Next Step: Tutorial
   :link: ../rasa-tutorial/



|

-------------------------------------------

.. _installation_guide:

Step-by-step Installation Guide
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. Install the Python development environment
---------------------------------------------

Check if your Python environment is already configured:

.. code-block:: bash

    $ python3 --version
    $ pip3 --version

If these packages are already installed, these commands should display version
numbers for each step, and you can skip to the next step.

Otherwise, proceed with the instructions below to install them.

.. tabs::

    .. tab:: Ubuntu

        Fetch the relevant packages using ``apt``, and install virtualenv using ``pip``.

        .. code-block:: bash

            $ sudo apt update
            $ sudo apt install python3-dev python3-pip

    .. tab:: macOS

        Install the `Homebrew <https://brew.sh>`_ package manager if you haven't already.

        Once you're done, you can install Python3.

        .. code-block:: bash

            $ brew update
            $ brew install python

    .. tab:: Windows

        .. raw:: html

            Make sure the Microsoft VC++ Compiler is installed, so python can compile
            any dependencies. You can get the compiler from <a class="reference external"
            href="https://visualstudio.microsoft.com/visual-cpp-build-tools/"
            target="_blank">Visual Studio</a>. Download the installer and select
            VC++ Build tools in the list.

        Install `Python 3 <https://www.python.org/downloads/windows/>`_ (64-bit version) for Windows.

        .. code-block:: bat

            C:\> pip3 install -U pip


2. Create a virtual environment (strongly recommended)
------------------------------------------------------

Tools like `virtualenv <https://virtualenv.pypa.io/en/latest/>`_ and `virtualenvwrapper <https://virtualenvwrapper.readthedocs.io/en/latest/>`_ provide isolated Python environments, which are cleaner than installing packages systemwide (as they prevent dependency conflicts). They also let you install packages without root privileges.

.. tabs::

    .. tab:: Ubuntu / macOS

        Create a new virtual environment by choosing a Python interpreter and making a ``./venv`` directory to hold it:

        .. code-block:: bash

            $ python3 -m venv --system-site-packages ./venv

        Activate the virtual environment:

        .. code-block:: bash

            $ source ./venv/bin/activate

    .. tab:: Windows

        Create a new virtual environment by choosing a Python interpreter and making a ``.\venv`` directory to hold it:

        .. code-block:: bat

            C:\> python3 -m venv --system-site-packages ./venv

        Activate the virtual environment:

        .. code-block:: bat

            C:\> .\venv\Scripts\activate


3. Install Rasa and Rasa X
--------------------------

.. tabs::

    .. tab:: Rasa and Rasa X

        To install both Rasa and Rasa X in one go:

        .. code-block:: bash

            $ pip install rasa-x --extra-index-url https://pypi.rasa.com/simple

    .. tab:: Rasa only

        If you just want to install Rasa without Rasa X:

        .. code-block:: bash

            $ pip install rasa

**Congratulations! You have successfully installed Rasa!**

You can now head over to the tutorial.

.. button::
   :text: Next Step: Tutorial
   :link: ../rasa-tutorial/

|

-------------------------------------------


.. _build_from_source:

Building from Source
~~~~~~~~~~~~~~~~~~~~

If you want to use the development version of Rasa, you can get it from GitHub:

.. code-block:: bash

    $ git clone https://github.com/RasaHQ/rasa.git
    $ cd rasa
    $ pip install -r requirements.txt
    $ pip install -e .

--------------------------------

.. _pipeline_dependencies:

NLU Pipeline Dependencies
~~~~~~~~~~~~~~~~~~~~~~~~~

Several NLU components have additional dependencies that need to
be installed separately.

Here, you will find installation instructions for each of them below.

How do I choose a pipeline?
---------------------------

The page on :ref:`choosing-a-pipeline` will help you pick the right pipeline
for your assistant.

I have decided on a pipeline. How do I install the dependencies for it?
-----------------------------------------------------------------------

When you install Rasa, the dependencies for the ``supervised_embeddings`` - TensorFlow
and sklearn_crfsuite get automatically installed. However, spaCy and MITIE need to be separately installed if you want to use pipelines containing components from those libraries.

.. admonition:: Just give me everything!

    If you don't mind the additional dependencies lying around, you can use
    this to install everything.

    You'll first need to clone the repository and then run the following
    command to install all the packages:

    .. code-block:: bash

        $ pip install -r alt_requirements/requirements_full.txt


Dependencies for spaCy
######################


For more information on spaCy, check out the `spaCy docs <https://spacy.io/usage/models>`_.

You can install it with the following commands:

.. code-block:: bash

    $ pip install rasa[spacy]
    $ python -m spacy download en_core_web_md
    $ python -m spacy link en_core_web_md en

This will install Rasa NLU as well as spacy and its language model
for the English language. We recommend using at least the
"medium" sized models (``_md``) instead of the spacy's
default small ``en_core_web_sm`` model. Small models require less
memory to run, but will somewhat reduce intent classification performance.

.. _install-mitie:

Dependencies for MITIE
######################

First, run

.. code-block:: bash

    $ pip install git+https://github.com/mit-nlp/MITIE.git
    $ pip install rasa[mitie]

and then download the
`MITIE models <https://github.com/mit-nlp/MITIE/releases/download/v0.4/MITIE-models-v0.2.tar.bz2>`_.
The file you need is ``total_word_feature_extractor.dat``. Save this
anywhere. If you want to use MITIE, you need to
tell it where to find this file (in this example it was saved in the
``data`` folder of the project directory).

.. warning::
    
    Mitie support is likely to be deprecated in a future release.
