.. _section_tutorial:

.. _tutorial:

Tutorial: A simple restaurant search bot
========================================

.. note:: See :ref:`section_migration` for how to clone your existing wit/LUIS/api.ai app.

As an example we'll use the domain of searching for restaurants. 
We'll start with an extremely simple model of those conversations. You can build up from there.

Let's assume that `anything` our bot's users say can be categorized into one of the following **intents**:

- ``greet``
- ``restaurant_search``
- ``thankyou``

Of course there are many ways our users might ``greet`` our bot: 

- `Hi!`
- `Hey there!`
- `Hello again :)`

And even more ways to say that you want to look for restaurants:

- `Do you know any good pizza places?`
- `I'm in the North of town and I want chinese food`
- `I'm hungry`

The first job of rasa NLU is to assign any given sentence to one of the **intent** categories: ``greet``, ``restaurant_search``, or ``thankyou``.

The second job is to label words like "Mexican" and "center" as ``cuisine`` and ``location`` **entities**, respectively. 
In this tutorial we'll build a model which does exactly that.

Preparing the Training Data
---------------------------

The training data is essential to develop chatbots. It should include texts to be interpreted and the structured data (intent/entities) we expect chatbots to convert the texts into. The best way to get training texts is from *real users*, and the best way to get the structured data is to `pretend to be the bot yourself <https://conversations.golastmile.com/put-on-your-robot-costume-and-be-the-minimum-viable-bot-yourself-3e48a5a59308#.d4tmdan68>`_. But to help get you started, we have some `data saved <https://github.com/RasaHQ/rasa_nlu/blob/master/data/examples/rasa/demo-rasa.json>`_.

Download the file (json format) and open it, and you'll see a list of training examples, each composed of ``"text"``, ``"intent"`` and ``"entities"``, as shown below. In your working directory, create a ``data`` folder, and copy this ``demo-rasa.json`` file there.

.. code-block:: json

    {
      "text": "hey", 
      "intent": "greet", 
      "entities": []
    }

.. code-block:: json

    {
      "text": "show me chinese restaurants", 
      "intent": "restaurant_search", 
      "entities": [
        {
          "start": 8, 
          "end": 15, 
          "value": "chinese", 
          "entity": "cuisine"
        }
      ]
    }

Hopefully the format is intuitive if you've read this far into the tutorial, for details see :ref:`section_dataformat`. Otherwise, the next section 'visualizing the training data' can help you better read, verify and/or modify the training data.

.. _visualizing-the-training-data:

Visualizing the Training Data
-----------------------------

It's always a good idea to `look` at your data before, during, and after training a model. 
Luckily, there's a `great tool <https://github.com/RasaHQ/rasa-nlu-trainer>`__ for creating training data in rasa's format.
- created by `@azazdeaz <https://github.com/azazdeaz>`_ - and it's also extremely helpful for inspecting and modifying existing data. 


For the demo data the output should look like this:

.. image:: _static/images/rasa_nlu_intent_gui.png


It is **strongly** recommended that you view your training data in the GUI before training.

.. _training_your_model:

Training Your Model
-------------------

Now we're going to create a configuration file. Make sure first that you've set up a backend, see :ref:`section_backends` .
Create a file called ``config_spacy.json`` or ``config_mitie.json``, depending on the pipeline selected, in your working directory which looks like this

 
.. literalinclude:: ../sample_configs/config_spacy.json
    :language: json

or if you've installed the MITIE backend instead:

 
.. literalinclude:: ../sample_configs/config_mitie.json
    :language: json

Now we can train a spacy model by running:

.. code-block:: console

    $ python -m rasa_nlu.train -c sample_configs/config_spacy.json

If you want to know more about the parameters, there is an overview of the :ref:`section_configuration`. After a few minutes,
rasa NLU will finish training, and you'll see a new folder named as 
``models/model_YYYYMMDD-HHMMSS`` with the timestamp when training finished.


Using Your Model
----------------

To run your trained model, pass the configuration value ``server_model_dirs`` when running the server:

.. code-block:: console

    $ python -m rasa_nlu.server -c sample_configs/config_spacy.json --server_model_dirs=./model_YYYYMMDD-HHMMSS

The passed model path is relative to the ``path`` configured in the configuration. More information about starting the server can be found in :ref:`section_http`.

You can then test our your new model by sending a request. Open a new tab/window on your terminal and run

.. code-block:: console

    $ curl -XPOST localhost:5000/parse -d '{"q":"I am looking for Chinese food"}' | python -mjson.tool

which should return 

.. code-block:: json

    {
        "text": "I am looking for Chinese food",
        "entities": [
            {
              "start": 8,
              "end": 15,
              "value": "chinese",
              "entity": "cuisine",
              "extractor": "ner_spacy"
            }
        ],
        "intent": {
            "confidence": 0.6485910906220309,
            "name": "restaurant_search"
        },
        "intent_ranking": [
            {
                "confidence": 0.6485910906220309,
                "name": "restaurant_search"
            },
            {
                "confidence": 0.14161531595656784,
                "name": "affirm"
            }
        ]
    }

If you are using the ``spacy_sklearn`` backend and the entities aren't found, don't panic!
This tutorial is just a toy example, with far too little training data to expect good performance.

rasa NLU will also print a ``confidence`` value for the intent classification. For models using spacy
intent classification this will be a probability. For MITIE models this is just a score, which **might be
greater than 1**.

You can use this to do some error handling in your chatbot (ex: asking the user again if the confidence is low)
and it's also helpful for prioritising which intents need more training data.

.. note::
    The output may contain other or less attributes, depending on the pipeline you are using. For
    example, the ``mitie`` pipeline doesn't include the ``"intent_ranking"`` whereas the ``spacy_sklearn``
    pipeline does.


With very little data, rasa NLU can in certain cases already generalise concepts, for example:


.. code-block:: console

    $ curl -XPOST localhost:5000/parse -d '{"q":"I want some italian"}' | python -mjson.tool
    {
        "text": "I want some italian",
        "entities": [
            {
              "end": 19,
              "entity": "cuisine",
              "start": 12,
              "value": "italian",
              "extrator": "ner_mitie"
            }
        ],
        "intent": {
            "confidence": 0.5192305466357352,
            "name": "restaurant_search"
        },
        "intent_ranking": [
            {
                "confidence": 0.5192305466357352,
                "name": "restaurant_search"
            },
            {
                "confidence": 0.2066287604378098,
                "name": "affirm"
            }
        ]
    }

even though there's nothing quite like this sentence in the examples used to train the model. 
To build a more robust app you will obviously want to use a lot more training data, so go and collect it!
