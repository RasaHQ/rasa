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
    python -m rasa_nlu.train -c sample_configs/config_composite_entities.yml \
        --data data/df-agent/training_data.json -o models \
        --fixed_model_name df-agent --project current --verbose

Then start the server with ``python -m rasa_nlu.server --debug --path models/``

Then running

.. code-block:: bash
    curl 'http://localhost:5000/parse?project=current&model=df-agent&q=show%20me%20details%20of%20the%202017%20kia%20rio'
    
    {
        "project": "current",
        "entities": [{
            "extractor": "ner_crf",
            "confidence": 0.9911352796662645,
            "end": 35,
            "processors": [
                "nested_entity_extractor"
            ],
            "value": {
                "car": {
                    "make": "kia",
                    "model": "rio",
                    "year": "2017"
                }
            },
            "entity": "car",
            "start": 23
        }],
        "intent": {
            "confidence": 0.966084897518158,
            "name": "request.details"
        },
        "text": "show me details of the 2017 kia rio",
        "model": "df-agent",
        "intent_ranking": [{
                "confidence": 0.966084897518158,
                "name": "request.details"
            },
            ...
        ]
    }
