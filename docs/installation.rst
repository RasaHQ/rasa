.. _installation:

Installation
============

Install Rasa Core to get started with the Rasa stack.

.. note::
    You can also get started without installing anything by using this

    .. button::
       :text: colab notebook 
       :link: https://colab.research.google.com/github/RasaHQ/rasa_core/blob/master/getting_started.ipynb

Install Rasa Core
-----------------
The recommended way to install Rasa Core is using pip:

.. copyable:: 

    pip install rasa_core

If you alredy have `rasa_core` installed and want to update it run:

.. code-block:: bash

    pip install -U rasa_core

Unless you've already got numpy & scipy installed, we highly recommend 
that you install and use `Anaconda <https://www.continuum.io\/downloads>`_.


If you want to use the bleeding edge version of Rasa use github + setup.py:

.. code-block:: bash

    git clone https://github.com/RasaHQ/rasa_core.git
    cd rasa_core
    pip install -r requirements.txt
    pip install -e .

.. note::
    If you want to change the Rasa Core code and want to run the tests or
    build the documentation, you need to install the development dependencies:

    .. code-block:: bash

        pip install -r dev-requirements.txt
        pip install -e .


Add Natural Language Understanding
----------------------------------

We use Rasa NLU for intent classification & entity extraction. To get it, run


.. code-block:: bash

    pip install rasa_nlu[tensorflow]

Full instructions can be found `here <https://rasa.com/docs/nlu/installation/>`_.

You can also use other NLU services like wit.ai, dialogflow, or LUIS. 
In fact, you don't need to use NLU at all, if your messaging app uses buttons
rather than free text.

Getting Started
---------------

To see your newly installed Rasa Core in action, head over to the 
:ref:`quickstart`.
