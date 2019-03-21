:desc: Create custom components to create additional features like sentiment
       analysis to integrate with open source bot framework Rasa Stack.

.. _section_customcomponents:

Custom Components
=================

You can create a custom Component to perform a specific task which NLU doesn't currently offer (for example, sentiment analysis).
Below is the specification of the :class:`rasa.nlu.components.Component` class with the methods you'll need to implement.

You can add a custom component to your pipeline by adding the module path.
So if you have a module called ``sentiment``
containing a ``SentimentAnalyzer`` class:

    .. code-block:: yaml

        pipeline:
        - name: "sentiment.SentimentAnalyzer"


Also be sure to read the section on the :ref:`section_component_lifecycle` .

To get started, you can use this skeleton that contains the most important
methods that you should implement:

.. literalinclude:: ../../tests/nlu/example_component.py
    :language: python
    :linenos:


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


.. include:: feedback.inc
