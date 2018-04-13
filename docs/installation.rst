.. _section_backends:

Installation
============

Rasa NLU itself doesn't have any external requirements,
but to do something useful with it you need to
install & configure a backend. Which backend you want to use is up to you.

Setting up Rasa NLU
~~~~~~~~~~~~~~~~~~~
The recommended way to install Rasa NLU is using pip:

.. code-block:: bash

    pip install rasa_nlu

If you want to use the bleeding edge version use github + setup.py:

.. code-block:: bash

    git clone https://github.com/RasaHQ/rasa_nlu.git
    cd rasa_nlu
    pip install -r requirements.txt
    pip install -e .

Rasa NLU allows you to use components to process your messages.
E.g. there is a component for intent classification and
there are several different components for entity recognition.
The different components have their own requirements. To get
you started quickly, this installation guide only installs
the basic requirements, you may need to install other
dependencies if you want to use certain components. When running
Rasa NLU it will check if all required dependencies are
installed and tell you if any are missing.

.. note::

    If you want to make sure you have the dependencies
    installed for any component you might ever need, and you
    don't mind the additional dependencies lying around, you can use

    .. code-block:: bash

        pip install -r alt_requirements/requirements_full.txt

    to install everything.


Best for most: spaCy + sklearn
------------------------------

Rasa NLU can run with a choice of backends, but for most users
a combination of spaCy and scikit-learn is the best option.

Installing spacy just requires (for more information
visit the `spacy docu <https://spacy.io/docs/usage/>`_):

.. code-block:: bash

    pip install rasa_nlu[spacy]
    python -m spacy download en_core_web_md
    python -m spacy link en_core_web_md en

This will install Rasa NLU as well as spacy and its language model
for the english language. We recommend using at least the
"medium" sized models (``_md``) instead of the spacy's
default small ``en_core_web_sm`` model. Small models require less 
memory to run, but will somewhat reduce intent classification performance.

.. note::

    Using spaCy as the backend for Rasa NLU is the **preferred option**.
    In older versions of Rasa NLU, MITIE was another supported backend.
    MITIE support has been deprecated as of version 0.12.
