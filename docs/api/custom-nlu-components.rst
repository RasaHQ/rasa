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


Also be sure to read the section on the :ref:`section_component_lifecycle`.

To get started, you can use this skeleton that contains the most important
methods that you should implement:

.. literalinclude:: ../../tests/nlu/example_component.py
    :language: python
    :linenos:

.. note::
    If you create a custom tokenizer you should have the following entry in your default dict.

    .. code-block:: none

        # add __CLS__ token to the end of the list of tokens
        "use_cls_token": False,

    To add the __CLS__ token you can use the function `add_cls_token()
    <https://github.com/RasaHQ/rasa/blob/master/rasa/nlu/tokenizers/tokenizer.py#L64/>`_.

.. note::
    If you create a custom tokenizer you should have the following entry in your default dict.

    .. code-block:: none

        # if True return a sequence of features (return vector has size
        # token-size x feature-dimension)
        # if False token-size will be equal to 1
        "return_sequence": False,

Component
^^^^^^^^^

.. autoclass:: rasa.nlu.components.Component

   .. automethod:: required_packages

   .. automethod:: create

   .. automethod:: provide_context

   .. automethod:: train

   .. automethod:: process

   .. automethod:: persist

   .. automethod:: prepare_partial_processing

   .. automethod:: partially_process

   .. automethod:: can_handle_language
