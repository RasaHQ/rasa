.. _section_http:

Using rasa NLU from python
====================================

Training Time
------------------------------------
For creating your models, follow the same instructions as non-python users.


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