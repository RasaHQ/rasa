.. _section_python:

Using rasa NLU from python
====================================

Training Time
------------------------------------
For creating your models, you can follow the same instructions as non-python users.
Or, you can train directly in python with a script like the following: 

.. code-block:: python

    from rasa_nlu.training_data import TrainingData
    from rasa_nlu.trainers.spacy_sklearn_trainer import SpacySklearnTrainer

    training_data = TrainingData('data/dataset.json', 'spacy_sklearn', 'en')
    trainer.train(training_data)
    trainer.persist('models/')


Prediction Time
-------------------------

You can call rasa NLU directly from your python script. 
You just have to instantiate either the SpacySklearnInterpreter or the MITIEInterpreter.
The ``metadata.json`` in your model dir contains the necessary info, so you can just do

.. code-block:: python

    from rasa_nlu.interpreters.spacy_sklearn_interpreter import SpacySklearnInterpreter
    metadata = json.loads(open('/path/to/metadata.json').read())
    interpreter = SpacySklearnInterpreter(**metadata)

You can then run:

.. code-block:: python

    interpreter.parse("The text I want to unterstand")

which returns the same ``dict`` as the HTTP api would (without emulation).