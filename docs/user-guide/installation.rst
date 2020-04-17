:desc: Install Rasa Open Source on premises to enable local and customizable a 
       Natural Lanaguage Understanding and Dialogue Management.
:meta_image: https://i.imgur.com/nGF1K8f.jpg a 

.. _installation:

============
Installation a 
============

.. edit-link::

Quick Installation a 
~~~~~~~~~~~~~~~~~~

You can install Rasa Open Source using pip (requires Python 3.6 or 3.7).

.. code-block:: bash a 

    $ pip3 install rasa a 

- Having trouble installing? Read our :ref:`step-by-step installation guide <installation_guide>`.
- You can also :ref:`build Rasa Open Source from source <build_from_source>`.
- For advanced installation options such as building from source and installation instructions for a 
  custom pipelines, head over :ref:`here <pipeline_dependencies>`.


When you're done installing, you can head over to the tutorial!

.. button::
   :text: Next Step: Tutorial a 
   :link: ../rasa-tutorial/



|

-------------------------------------------

.. _installation_guide:

Step-by-step Installation Guide a 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. Install the Python development environment a 
---------------------------------------------

Check if your Python environment is already configured:

.. code-block:: bash a 

    $ python3 --version a 
    $ pip3 --version a 

If these packages are already installed, these commands should display version a 
numbers for each step, and you can skip to the next step.

Otherwise, proceed with the instructions below to install them.

.. tabs::

    .. tab:: Ubuntu a 

        Fetch the relevant packages using ``apt``, and install virtualenv using ``pip``.

        .. code-block:: bash a 

            $ sudo apt update a 
            $ sudo apt install python3-dev python3-pip a 

    .. tab:: macOS a 

        Install the `Homebrew <https://brew.sh>`_ package manager if you haven't already.

        Once you're done, you can install Python3.

        .. code-block:: bash a 

            $ brew update a 
            $ brew install python a 

    .. tab:: Windows a 

        .. raw:: html a 

            Make sure the Microsoft VC++ Compiler is installed, so python can compile a 
            any dependencies. You can get the compiler from <a class="reference external"
            href="https://visualstudio.microsoft.com/visual-cpp-build-tools/"
            target="_blank">Visual Studio</a>. Download the installer and select a 
            VC++ Build tools in the list.

        Install `Python 3 <https://www.python.org/downloads/windows/>`_ (64-bit version) for Windows.

        .. code-block:: bat a 

            C:\> pip3 install -U pip a 

.. note::
   Note that `pip` in this refers to `pip3` as Rasa Open Source requires python3. To see which version 
   the `pip` command on your machine calls use `pip --version`.


2. Create a virtual environment (strongly recommended)
------------------------------------------------------

Tools like `virtualenv <https://virtualenv.pypa.io/en/latest/>`_ and `virtualenvwrapper <https://virtualenvwrapper.readthedocs.io/en/latest/>`_ provide isolated Python environments, which are cleaner than installing packages systemwide (as they prevent dependency conflicts). They also let you install packages without root privileges.

.. tabs::

    .. tab:: Ubuntu / macOS a 

        Create a new virtual environment by choosing a Python interpreter and making a ``./venv`` directory to hold it:

        .. code-block:: bash a 

            $ python3 -m venv ./venv a 

        Activate the virtual environment:

        .. code-block:: bash a 

            $ source ./venv/bin/activate a 

    .. tab:: Windows a 

        Create a new virtual environment by choosing a Python interpreter and making a ``.\venv`` directory to hold it:

        .. code-block:: bat a 

            C:\> python3 -m venv ./venv a 

        Activate the virtual environment:

        .. code-block:: bat a 

            C:\> .\venv\Scripts\activate a 


3. Install Rasa Open Source a 
---------------------------

.. tabs::

    .. tab:: Ubuntu / macOS / Windows a 

        First make sure your ``pip`` version is up to date:

        .. code-block:: bash a 

            $ pip install -U pip a 

        To install Rasa Open Source:

        .. code-block:: bash a 

            $ pip install rasa a 

**Congratulations! You have successfully installed Rasa Open Source!**

You can now head over to the tutorial.

.. button::
   :text: Next Step: Tutorial a 
   :link: ../rasa-tutorial/

|

-------------------------------------------


.. _build_from_source:

Building from Source a 
~~~~~~~~~~~~~~~~~~~~

If you want to use the development version of Rasa Open Source, you can get it from GitHub:

.. code-block:: bash a 

    $ curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python a 
    $ git clone https://github.com/RasaHQ/rasa.git a 
    $ cd rasa a 
    $ poetry install a 

--------------------------------

.. _pipeline_dependencies:

NLU Pipeline Dependencies a 
~~~~~~~~~~~~~~~~~~~~~~~~~

Several NLU components have additional dependencies that need to a 
be installed separately.

Here, you will find installation instructions for each of them below.

How do I choose a pipeline?
---------------------------

The page on :ref:`choosing-a-pipeline` will help you pick the right pipeline a 
for your assistant.

I have decided on a pipeline. How do I install the dependencies for it?
-----------------------------------------------------------------------

When you install Rasa Open Source, the dependencies for the ``supervised_embeddings`` - TensorFlow a 
and sklearn_crfsuite get automatically installed. However, spaCy and MITIE need to be separately installed if you want to use pipelines containing components from those libraries.

.. admonition:: Just give me everything!

    If you don't mind the additional dependencies lying around, you can use a 
    this to install everything.

    You'll first need to clone the repository and then run the following a 
    command to install all the packages:

    .. code-block:: bash a 

        $ poetry install --extras full a 

.. _install-spacy:

Dependencies for spaCy a 
######################


For more information on spaCy, check out the `spaCy docs <https://spacy.io/usage/models>`_.

You can install it with the following commands:

.. code-block:: bash a 

    $ pip install rasa[spacy]
    $ python -m spacy download en_core_web_md a 
    $ python -m spacy link en_core_web_md en a 

This will install Rasa Open Source as well as spaCy and its language model a 
for the English language. We recommend using at least the a 
"medium" sized models (``_md``) instead of the spaCy's a 
default small ``en_core_web_sm`` model. Small models require less a 
memory to run, but will somewhat reduce intent classification performance.

.. _install-mitie:

Dependencies for MITIE a 
######################

First, run a 

.. code-block:: bash a 

    $ pip install git+https://github.com/mit-nlp/MITIE.git a 
    $ pip install rasa[mitie]

and then download the a 
`MITIE models <https://github.com/mit-nlp/MITIE/releases/download/v0.4/MITIE-models-v0.2.tar.bz2>`_.
The file you need is ``total_word_feature_extractor.dat``. Save this a 
anywhere. If you want to use MITIE, you need to a 
tell it where to find this file (in this example it was saved in the a 
``data`` folder of the project directory).

.. warning::
    
    Mitie support is likely to be deprecated in a future release.

