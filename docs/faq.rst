.. _section_faq:

Frequently Asked Questions
==========================

How many training examples do I need?
-------------------------------------
Unfortunately, there is no cookie-cutter answer to this question. It depends on your intents and your entities.

If you have intents that are easily confusable, you will need more training data. Accordingly, as you add more
intents, you also want to add more training examples for each intent. If you quickly write 20-30 unique expressions for
each intent, you should be good for the beginning.

The same holds true for entities. the number of training examples you will need depends on how closely related your different entity types are and how clearly
entities are distinguishable from non-entities in your use case.

To assess your model's performance, :ref:`run the server and manually test some messages <tutorial_using_your_model>`
, or use the :ref:`evaluation script <section_evaluation>`.



Does it run with python 3?
--------------------------
Yes it does, rasa NLU supports python 2.7 as well as python 3.5 and 3.6. If there are any issues with a specific python version, feel free to create an issue or directly provide a fix.

Which languages are supported?
------------------------------
There is a list containing all officialy supported languages :ref:`here <section_languages>`. Nevertheless, there are
others working on adding more languages, feel free to have a look at the `github issues <https://github.com/RasaHQ/rasa_nlu/issues>`_
section or the `gitter chat <https://gitter.im/RasaHQ/rasa_nlu>`_.

.. _section_faq_version:

Which version of rasa NLU am I running?
---------------------------------------
To find out which rasa version you are running, you can execute

.. code-block:: bash

   python -c "import rasa_nlu; print(rasa_nlu.__version__);"

If you are using a virtual environment to run your python code, make sure you are using the correct python to execute the above code.

Why am I getting an ``UndefinedMetricWarning``?
-----------------------------------------------
The complete warning is: ``UndefinedMetricWarning: F-score is ill-defined and being set to 0.0 in labels with no predicted samples.``
The warning is a result of a lack of training data. During the training the dataset will be splitted multiple times, if there are to few training samples for any of the intents, the splitting might result in splits that do not contain any examples for this intent.

Hence, the solution is to add more training samples. As this is only a warning, training will still succeed, but the resulting models predictions might be weak on the intents where you are lacking training data.  


I have an issue, can you help me?
---------------------------------
We'd love to help you. If you are unsure if your issue is related to your setup, you should state your problem in the `gitter chat <https://gitter.im/RasaHQ/rasa_nlu>`_.
If you found an issue with the framework, please file a report on `github issues <https://github.com/RasaHQ/rasa_nlu/issues>`_
including all the information needed to reproduce the problem.

.. toctree::
   :maxdepth: 1
