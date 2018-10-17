:desc: Using composite entities (like DialogFlow) in Rasa NLU 

.. _section_compositeentities:

Composite Entities
==================

Generating
----------

Copy your DialogFlow agent ``df-agent`` into the data folder

Run ``python transfer_from_dialogflow_to_rasa.py data/df-agent``
to generate the training data as Rasa needs it inside the df-agent folder.

Then to generate your model:

.. code-block:: bash
    python -m rasa_nlu.train -c config.yml \
        --data data/df-agent/training_data.json -o models \
        --fixed_model_name df-agent --project current --verbose

Then you may have to move the model from ``models/current/df-agent`` to ``models/current``

Then start the server with ``python -m rasa_nlu.server --debug --path models/current``

Then running

.. code-block:: bash
    curl 'http://localhost:5000/parse?q=show%20me%20details%20of%20the%202017%20kia%20rio'
    
    {
    "intent": {
        "name": null,
        "confidence": 1.0
    },
    "entities": [],
    "text": "show me details of the 2017 kia rio",
    "project": "default",
    "model": "fallback"
    }%
