.. _installation:

Installation
============

Install Rasa Core to get started with the Rasa stack.

.. note::

    You can also get started without installing anything by going
    to the :ref:`quickstart`


Install Rasa Core
-----------------

Stable (Most recent release)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The recommended way to install Rasa Core is using pip:

.. copyable:: 

    pip install rasa_core


If you alredy have `rasa_core` installed and want to update it run:

.. copyable:: 

    pip install -U rasa_core

Unless you've already got numpy & scipy installed, we highly recommend 
that you install and use `Anaconda <https://www.continuum.io\/downloads>`_.

.. note::

    If you want to run custom action code, please also take a look at
    :ref:`customactions`. You'll need to install Rasa Core to train and
    use the model and ``rasa_core_sdk`` to develop your custom action code.


Latest (Most recent github)
~~~~~~~~~~~~~~~~~~~~~~~~~~~
If you want to use the bleeding edge version of Rasa use github + setup.py:

.. code-block:: bash

    git clone https://github.com/RasaHQ/rasa_core.git
    cd rasa_core
    pip install -r requirements.txt
    pip install -e .

Development (github & development dependencies)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you want to change the Rasa Core code and want to run the tests or
build the documentation, you need to install the development dependencies:

.. code-block:: bash

    pip install -r dev-requirements.txt
    pip install -e .


Add Natural Language Understanding
----------------------------------

We use Rasa NLU for intent classification & entity extraction. To get it, run:

.. code-block:: bash

    pip install rasa_nlu[tensorflow]

Full instructions can be found
`in the NLU documentation <https://rasa.com/docs/nlu/installation/>`_.

You can also use other NLU services like wit.ai, dialogflow, or LUIS. 
In fact, you don't need to use NLU at all, if your messaging app uses buttons
rather than free text.

Build your first Rasa assistant!
--------------------------------
After following the quickstart and installing Rasa Core, the next step is to 
build your first Rasa assistant yourself! To get you started, we have prepared a 
Rasa Stack starter-pack which has all the files you need to build your first custom 
chatbot. On top of that, the starter-pack includes a training data set ready 
for you to use.

Click the linke below to get the Rasa Stack starter-pack:
	
`Rasa Stack starter-pack <https://github.com/RasaHQ/starter-pack-rasa-stack>`_
	
Let us know how you are getting on! If you have any questions about the starter-pack or 
using Rasa Stack in general, post your questions on `Rasa Community Forum <https://forum.rasa.com>`_!


.. include:: feedback.inc

.. raw:: html
   :file: livechat.html
