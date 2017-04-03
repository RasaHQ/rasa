.. _section_python:

Using rasa NLU from python
==========================
Apart from running rasa NLU as a HTTP server you can use it directly in your python program.
Rasa NLU supports both Python 2 and 3.

Training Time
-------------
For creating your models, you can follow the same instructions as non-python users.
Or, you can train directly in python with a script like the following (using spacy):

.. testcode::

    from rasa_nlu.converters import load_data
    from rasa_nlu.config import RasaNLUConfig
    from rasa_nlu.model import Trainer

    training_data = load_data('data/examples/rasa/demo-rasa.json', 'en')
    trainer = Trainer(RasaNLUConfig("config_spacy.json"))
    trainer.train(training_data)
    model_directory = trainer.persist('./models/')  # Returns the directory the model is stored in

Prediction Time
---------------

You can call rasa NLU directly from your python script. To do so, you need to load the metadata of
your model and instantiate an interpreter. The ``metadata.json`` in your model dir contains the
necessary info to recover your model:

.. testcode::

    from rasa_nlu.model import Metadata, Interpreter

    metadata = Metadata.load(model_directory)   # where model_directory points to the folder the model is persisted in
    interpreter = Interpreter.load(metadata, RasaNLUConfig("config_spacy.json"))

You can then use the loaded interpreter to parse text:

.. testcode::

    interpreter.parse(u"The text I want to understand")

which returns the same ``dict`` as the HTTP api would (without emulation).