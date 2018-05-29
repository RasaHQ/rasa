.. _section_customcomponents:

Custom Components
=================

You can create a custom Component to perform a specific task which NLU doesn't currently offer (e.g. sentiment analysis).
A glimpse into the code of ``rasa_nlu.components.Component`` will reveal
which functions need to be implemented to create a new component.
You can add these to your pipeline by adding the module path to your pipeline, e.g. if you have a module called ``sentiment``
containing a ``SentimentAnalyzer`` class:

    .. code-block:: yaml

        pipeline:
        - name: "sentiment.SentimentAnalyzer"


Also be sure to read the section on the `:ref:section_component_lifecycle`_.

