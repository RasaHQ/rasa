:desc: Install Rasa Open Source on premises to enable local and customizable a
       Natural Lanaguage Understanding and Dialogue Management. a
:meta_image: https://i.imgur.com/nGF1K8f.jpg a
 a
.. _installation: a
 a
============ a
Installation a
============ a
 a
.. edit-link:: a
 a
Quick Installation a
~~~~~~~~~~~~~~~~~~ a
 a
You can install Rasa Open Source using pip (requires Python 3.6 or 3.7). a
 a
.. code-block:: bash a
 a
    $ pip3 install rasa a
 a
- Having trouble installing? Read our :ref:`step-by-step installation guide <installation_guide>`. a
- You can also :ref:`build Rasa Open Source from source <build_from_source>`. a
- For advanced installation options such as building from source and installation instructions for a
  custom pipelines, head over :ref:`here <pipeline_dependencies>`. a
 a
 a
When you're done installing, you can head over to the tutorial! a
 a
.. button:: a
   :text: Next Step: Tutorial a
   :link: ../rasa-tutorial/ a
 a
 a
 a
| a
 a
------------------------------------------- a
 a
.. _installation_guide: a
 a
Step-by-step Installation Guide a
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ a
 a
1. Install the Python development environment a
--------------------------------------------- a
 a
Check if your Python environment is already configured: a
 a
.. code-block:: bash a
 a
    $ python3 --version a
    $ pip3 --version a
 a
If these packages are already installed, these commands should display version a
numbers for each step, and you can skip to the next step. a
 a
Otherwise, proceed with the instructions below to install them. a
 a
.. tabs:: a
 a
    .. tab:: Ubuntu a
 a
        Fetch the relevant packages using ``apt``, and install virtualenv using ``pip``. a
 a
        .. code-block:: bash a
 a
            $ sudo apt update a
            $ sudo apt install python3-dev python3-pip a
 a
    .. tab:: macOS a
 a
        Install the `Homebrew <https://brew.sh>`_ package manager if you haven't already. a
 a
        Once you're done, you can install Python3. a
 a
        .. code-block:: bash a
 a
            $ brew update a
            $ brew install python a
 a
    .. tab:: Windows a
 a
        .. raw:: html a
 a
            Make sure the Microsoft VC++ Compiler is installed, so python can compile a
            any dependencies. You can get the compiler from <a class="reference external" a
            href="https://visualstudio.microsoft.com/visual-cpp-build-tools/" a
            target="_blank">Visual Studio</a>. Download the installer and select a
            VC++ Build tools in the list. a
 a
        Install `Python 3 <https://www.python.org/downloads/windows/>`_ (64-bit version) for Windows. a
 a
        .. code-block:: bat a
 a
            C:\> pip3 install -U pip a
 a
.. note:: a
   Note that `pip` in this refers to `pip3` as Rasa Open Source requires python3. To see which version  a
   the `pip` command on your machine calls use `pip --version`. a
 a
 a
2. Create a virtual environment (strongly recommended) a
------------------------------------------------------ a
 a
Tools like `virtualenv <https://virtualenv.pypa.io/en/latest/>`_ and `virtualenvwrapper <https://virtualenvwrapper.readthedocs.io/en/latest/>`_ provide isolated Python environments, which are cleaner than installing packages systemwide (as they prevent dependency conflicts). They also let you install packages without root privileges. a
 a
.. tabs:: a
 a
    .. tab:: Ubuntu / macOS a
 a
        Create a new virtual environment by choosing a Python interpreter and making a ``./venv`` directory to hold it: a
 a
        .. code-block:: bash a
 a
            $ python3 -m venv ./venv a
 a
        Activate the virtual environment: a
 a
        .. code-block:: bash a
 a
            $ source ./venv/bin/activate a
 a
    .. tab:: Windows a
 a
        Create a new virtual environment by choosing a Python interpreter and making a ``.\venv`` directory to hold it: a
 a
        .. code-block:: bat a
 a
            C:\> python3 -m venv ./venv a
 a
        Activate the virtual environment: a
 a
        .. code-block:: bat a
 a
            C:\> .\venv\Scripts\activate a
 a
 a
3. Install Rasa Open Source a
--------------------------- a
 a
.. tabs:: a
 a
    .. tab:: Ubuntu / macOS / Windows a
 a
        First make sure your ``pip`` version is up to date: a
 a
        .. code-block:: bash a
 a
            $ pip install -U pip a
 a
        To install Rasa Open Source: a
 a
        .. code-block:: bash a
 a
            $ pip install rasa a
 a
**Congratulations! You have successfully installed Rasa Open Source!** a
 a
You can now head over to the tutorial. a
 a
.. button:: a
   :text: Next Step: Tutorial a
   :link: ../rasa-tutorial/ a
 a
| a
 a
------------------------------------------- a
 a
 a
.. _build_from_source: a
 a
Building from Source a
~~~~~~~~~~~~~~~~~~~~ a
 a
If you want to use the development version of Rasa Open Source, you can get it from GitHub: a
 a
.. code-block:: bash a
 a
    $ curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python a
    $ git clone https://github.com/RasaHQ/rasa.git a
    $ cd rasa a
    $ poetry install a
 a
-------------------------------- a
 a
.. _pipeline_dependencies: a
 a
NLU Pipeline Dependencies a
~~~~~~~~~~~~~~~~~~~~~~~~~ a
 a
Several NLU components have additional dependencies that need to a
be installed separately. a
 a
Here, you will find installation instructions for each of them below. a
 a
How do I choose a pipeline? a
--------------------------- a
 a
The page on :ref:`choosing-a-pipeline` will help you pick the right pipeline a
for your assistant. a
 a
I have decided on a pipeline. How do I install the dependencies for it? a
----------------------------------------------------------------------- a
 a
When you install Rasa Open Source, the dependencies for the ``supervised_embeddings`` - TensorFlow a
and sklearn_crfsuite get automatically installed. However, spaCy and MITIE need to be separately installed if you want to use pipelines containing components from those libraries. a
 a
.. admonition:: Just give me everything! a
 a
    If you don't mind the additional dependencies lying around, you can use a
    this to install everything. a
 a
    You'll first need to clone the repository and then run the following a
    command to install all the packages: a
 a
    .. code-block:: bash a
 a
        $ poetry install --extras full a
 a
.. _install-spacy: a
 a
Dependencies for spaCy a
###################### a
 a
 a
For more information on spaCy, check out the `spaCy docs <https://spacy.io/usage/models>`_. a
 a
You can install it with the following commands: a
 a
.. code-block:: bash a
 a
    $ pip install rasa[spacy] a
    $ python -m spacy download en_core_web_md a
    $ python -m spacy link en_core_web_md en a
 a
This will install Rasa Open Source as well as spaCy and its language model a
for the English language. We recommend using at least the a
"medium" sized models (``_md``) instead of the spaCy's a
default small ``en_core_web_sm`` model. Small models require less a
memory to run, but will somewhat reduce intent classification performance. a
 a
.. _install-mitie: a
 a
Dependencies for MITIE a
###################### a
 a
First, run a
 a
.. code-block:: bash a
 a
    $ pip install git+https://github.com/mit-nlp/MITIE.git a
    $ pip install rasa[mitie] a
 a
and then download the a
`MITIE models <https://github.com/mit-nlp/MITIE/releases/download/v0.4/MITIE-models-v0.2.tar.bz2>`_. a
The file you need is ``total_word_feature_extractor.dat``. Save this a
anywhere. If you want to use MITIE, you need to a
tell it where to find this file (in this example it was saved in the a
``data`` folder of the project directory). a
 a
.. warning:: a
     a
    Mitie support is likely to be deprecated in a future release. a
 a