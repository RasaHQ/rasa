.. _section_python:

Using rasa NLU from python
==========================

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

You can call rasa NLU directly from your python script. 
You just have to instantiate either the SpacySklearnInterpreter or the MITIEInterpreter.
The ``metadata.json`` in your model dir contains the necessary info, so you can just do

.. testcode::

    from rasa_nlu.model import Metadata, Interpreter

    metadata = Metadata.load(model_directory)   # where model_directory points to the folder the model is persisted in
    interpreter = Interpreter.load(metadata, RasaNLUConfig("config_spacy.json"))

You can then run:

.. testcode::

    interpreter.parse(u"The text I want to understand")

which returns the same ``dict`` as the HTTP api would (without emulation).