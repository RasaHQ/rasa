.. _section_configuration:

Configuration
=============

You can provide options to Rasa NLU through:

- a yaml-formatted config file
- environment variables
- command line arguments

Environment variables override options in your config file, 
and command line args will override any options specified elsewhere.
Environment variables are capitalised and prefixed with ``RASA_``, 
so the option ``pipeline`` is specified with the ``RASA_PIPELINE`` env var.

Default
-------
Here is an example model configuration:

.. literalinclude:: ../sample_configs/config_crf.yml
    :language: yaml

As you can see, there are a couple of top-level configuration keys, like
``language`` and ``pipeline`` - but most of the configuration is component
specific.

Explanations for the configuration keys of the different components are part
of the :ref:`section_pipeline`.

Options
-------
A short explanation and examples for each configuration value.

pipeline
~~~~~~~~

:Type: ``str`` or ``[dict]``
:Examples:
    using a pipeline template (predefined set of components with default
    parameters):

    .. code-block:: yaml

        pipeline: "spacy_sklearn"

    or alternatively specifying the components and paremters:

    .. code-block:: yaml

        pipeline:
        - name: "nlp_spacy"
          model: "en"               # parameter of the spacy component
        - name: "ner_synonyms"

:Description:
    The pipeline used for training. Can either be a template
    (passing a string) or a list of components (array) and there
    configuration values. For all available templates,
    see :ref:`section_pipeline`. The component specific parameters
    are listed there as well.

language
~~~~~~~~

:Type: ``str``
:Examples:

    .. code-block:: yaml

        language: "en"

:Description:
    Language the model is trained in. Underlying word vectors
    will be loaded by using this language. There is more info
    about available languages in :ref:`section_languages`.
