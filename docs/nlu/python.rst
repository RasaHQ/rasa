:desc: Configure the Python API for serverless NLP using open source
       chatbot framework Rasa Stack.

.. _section_python:

Python API
==========
Apart from running Rasa NLU as a HTTP server you can use it directly in your python program.
Rasa NLU supports python 3.5, 3.6 and 3.7 (supported for python 2.7 up until version 0.14).

Training a Model
----------------
For creating your models, you can follow the same instructions as non-python users.
Or, you can train directly in python with a script like the following (using spacy):

.. testcode::

    from rasa.nlu.training_data import load_data
    from rasa.nlu.model import Trainer
    from rasa.nlu import config

    training_data = load_data('data/examples/rasa/demo-rasa.json')
    trainer = Trainer(config.load("sample_configs/config_pretrained_embeddings_spacy.yml"))
    trainer.train(training_data)
    model_directory = trainer.persist('./projects/default/')  # Returns the directory the model is stored in

Using a Model to Make Predictions
---------------------------------

You can call Rasa NLU directly from your python script. To do so, you need to load the metadata of
your model and instantiate an interpreter. The ``metadata.json`` in your model dir contains the
necessary information to reconstruct your model:

.. testcode::

    from rasa.nlu.model import Interpreter

    # where model_directory points to the model folder
    interpreter = Interpreter.load(model_directory)

You can then use the loaded interpreter to parse text:

.. testcode::

    interpreter.parse(u"The text I want to understand")

which returns the same data as the ``/parse`` endpoint of the :ref:`section_http` .

Reducing Memory Use When Loading Multiple Models
------------------------------------------------

If multiple models are created, it is reasonable to share components between the different models. E.g.
the ``'SpacyNLP'`` component, which is used by every pipeline that wants to have access to the spacy word vectors,
can be cached to avoid storing the large word vectors more than once in main memory. To use the caching,
a ``ComponentBuilder`` should be passed when loading and training models.

Here is a short example on how to create a component builder,
which can be reused to train and run multiple models.
To train a model:

.. testcode::

    from rasa.nlu.training_data import load_data
    from rasa.nlu import config
    from rasa.nlu.components import ComponentBuilder
    from rasa.nlu.model import Trainer

    builder = ComponentBuilder(use_cache=True)      # will cache components between pipelines (where possible)

    training_data = load_data('data/examples/rasa/demo-rasa.json')
    trainer = Trainer(config.load("sample_configs/config_pretrained_embeddings_spacy.yml"), builder)
    trainer.train(training_data)
    model_directory = trainer.persist('./projects/default/')  # Returns the directory the model is stored in

The same builder can be used to load a model (can be a totally different one). The builder only caches components that are safe to be shared between models. Here is a short example on how to use the builder when loading models:

.. testcode::

    from rasa.nlu.model import Interpreter
    from rasa.nlu import config

    # For simplicity we will load the same model twice, usually you would want to use the metadata of
    # different models

    interpreter = Interpreter.load(model_directory, builder)     # to use the builder, pass it as an arg when loading the model
    # the clone will share resources with the first model, as long as the same builder is passed!
    interpreter_clone = Interpreter.load(model_directory, builder)

Important Classes
-----------------

Config
^^^^^^

.. automodule:: rasa.nlu.config

   .. autofunction:: load


.. autoclass:: rasa.nlu.config.RasaNLUModelConfig

   .. automethod:: __init__


Interpreter
^^^^^^^^^^^

.. autoclass:: rasa.nlu.model.Interpreter


   .. automethod:: load

Metadata
^^^^^^^^

.. autoclass:: rasa.nlu.model.Metadata


   .. automethod:: load




ComponentBuilder
^^^^^^^^^^^^^^^^

.. autoclass:: rasa.nlu.components.ComponentBuilder


   .. automethod:: load_component


.. include:: feedback.inc
