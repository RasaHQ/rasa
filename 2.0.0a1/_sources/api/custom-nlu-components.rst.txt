:desc: Create custom components to create additional features like sentiment
       analysis to integrate with open source bot framework Rasa.

.. _custom-nlu-components:

Custom NLU Components
=====================

.. edit-link::

You can create a custom component to perform a specific task which NLU doesn't currently offer (for example, sentiment analysis).
Below is the specification of the :class:`rasa.nlu.components.Component` class with the methods you'll need to implement.

.. note::
    There is a detailed tutorial on building custom components `here
    <https://blog.rasa.com/enhancing-rasa-nlu-with-custom-components/>`_.


You can add a custom component to your pipeline by adding the module path.
So if you have a module called ``sentiment``
containing a ``SentimentAnalyzer`` class:

    .. code-block:: yaml

        pipeline:
        - name: "sentiment.SentimentAnalyzer"


Also be sure to read the section on the :ref:`component-lifecycle`.

To get started, you can use this skeleton that contains the most important
methods that you should implement:

.. literalinclude:: ../../tests/nlu/example_component.py
    :language: python
    :linenos:

.. note::
    If you create a custom tokenizer you should implement the methods of ``rasa.nlu.tokenizers.tokenizer.Tokenizer``.
    The ``train`` and ``process`` methods are already implemented and you simply need to overwrite the ``tokenize``
    method.

.. note::
    If you create a custom featurizer you can return two different kind of features: sequence features and sentence
    features. The sequence features are a matrix of size ``(number-of-tokens x feature-dimension)``, e.g.
    the matrix contains a feature vector for every token in the sequence.
    The sentence features are represented by a matrix of size ``(1 x feature-dimension)``.

Component
^^^^^^^^^

.. autoclass:: rasa.nlu.components.Component

   .. automethod:: required_components

   .. automethod:: required_packages

   .. automethod:: create

   .. automethod:: provide_context

   .. automethod:: train

   .. automethod:: process

   .. automethod:: persist

   .. automethod:: prepare_partial_processing

   .. automethod:: partially_process

   .. automethod:: can_handle_language
