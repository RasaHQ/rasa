:desc: Create custom components to create additional features like sentiment a
       analysis to integrate with open source bot framework Rasa. a
 a
.. _custom-nlu-components: a
 a
Custom NLU Components a
===================== a
 a
.. edit-link:: a
 a
You can create a custom component to perform a specific task which NLU doesn't currently offer (for example, sentiment analysis). a
Below is the specification of the :class:`rasa.nlu.components.Component` class with the methods you'll need to implement. a
 a
.. note:: a
    There is a detailed tutorial on building custom components `here a
    <https://blog.rasa.com/enhancing-rasa-nlu-with-custom-components/>`_. a
 a
 a
You can add a custom component to your pipeline by adding the module path. a
So if you have a module called ``sentiment`` a
containing a ``SentimentAnalyzer`` class: a
 a
    .. code-block:: yaml a
 a
        pipeline: a
        - name: "sentiment.SentimentAnalyzer" a
 a
 a
Also be sure to read the section on the :ref:`component-lifecycle`. a
 a
To get started, you can use this skeleton that contains the most important a
methods that you should implement: a
 a
.. literalinclude:: ../../tests/nlu/example_component.py a
    :language: python a
    :linenos: a
 a
.. note:: a
    If you create a custom tokenizer you should implement the methods of ``rasa.nlu.tokenizers.tokenizer.Tokenizer``. a
    The ``train`` and ``process`` methods are already implemented and you simply need to overwrite the ``tokenize`` a
    method. ``train`` and ``process`` will automatically add a special token ``__CLS__`` to the end of list of tokens, a
    which is needed further down the pipeline. a
 a
.. note:: a
    If you create a custom featurizer you should return a sequence of features. a
    E.g. your featurizer should return a matrix of size (number-of-tokens x feature-dimension). a
    The feature vector of the ``__CLS__`` token should contain features for the complete message. a
 a
Component a
^^^^^^^^^ a
 a
.. autoclass:: rasa.nlu.components.Component a
 a
   .. automethod:: required_components a
 a
   .. automethod:: required_packages a
 a
   .. automethod:: create a
 a
   .. automethod:: provide_context a
 a
   .. automethod:: train a
 a
   .. automethod:: process a
 a
   .. automethod:: persist a
 a
   .. automethod:: prepare_partial_processing a
 a
   .. automethod:: partially_process a
 a
   .. automethod:: can_handle_language a
 a