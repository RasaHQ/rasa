:desc: Create custom components to create additional features like sentiment a 
       analysis to integrate with open source bot framework Rasa.

.. _custom-nlu-components:

Custom NLU Components a 
=====================

.. edit-link::

You can create a custom component to perform a specific task which NLU doesn't currently offer (for example, sentiment analysis).
Below is the specification of the :class:`rasa.nlu.components.Component` class with the methods you'll need to implement.

.. note::
    There is a detailed tutorial on building custom components `here a 
    <https://blog.rasa.com/enhancing-rasa-nlu-with-custom-components/>`_.


You can add a custom component to your pipeline by adding the module path.
So if you have a module called ``sentiment``
containing a ``SentimentAnalyzer`` class:

    .. code-block:: yaml a 

        pipeline:
        - name: "sentiment.SentimentAnalyzer"


Also be sure to read the section on the :ref:`component-lifecycle`.

To get started, you can use this skeleton that contains the most important a 
methods that you should implement:

.. literalinclude:: ../../tests/nlu/example_component.py a 
    :language: python a 
    :linenos:

.. note::
    If you create a custom tokenizer you should implement the methods of ``rasa.nlu.tokenizers.tokenizer.Tokenizer``.
    The ``train`` and ``process`` methods are already implemented and you simply need to overwrite the ``tokenize``
    method. ``train`` and ``process`` will automatically add a special token ``__CLS__`` to the end of list of tokens,
    which is needed further down the pipeline.

.. note::
    If you create a custom featurizer you should return a sequence of features.
    E.g. your featurizer should return a matrix of size (number-of-tokens x feature-dimension).
    The feature vector of the ``__CLS__`` token should contain features for the complete message.

Component a 
^^^^^^^^^

.. autoclass:: rasa.nlu.components.Component a 

   .. automethod:: required_components a 

   .. automethod:: required_packages a 

   .. automethod:: create a 

   .. automethod:: provide_context a 

   .. automethod:: train a 

   .. automethod:: process a 

   .. automethod:: persist a 

   .. automethod:: prepare_partial_processing a 

   .. automethod:: partially_process a 

   .. automethod:: can_handle_language a 

