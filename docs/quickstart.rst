.. _section_quickstart:

.. _tutorial:

Quickstart
==========


As an example we'll start a new project covering the domain
of searching for restaurants. We'll start with an extremely simple
model of those conversations. You can build up from there.

Let's assume that `anything` our users say can be
categorized into one of the following **intents**:

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

The first job of Rasa NLU is to assign any given sentence to one of the **intent** categories: ``greet``, ``restaurant_search``, or ``thankyou``.

The second job is to label words like "chinese" and "North" as
``cuisine`` and ``location`` **entities**, respectively.
In this tutorial we'll build a model which does exactly that.

Preparing the Training Data
---------------------------

Training data is essential for developing chatbots and voice apps. 
The data is just a list of messages that you expect to receive, annotated with 
the intent and entities Rasa NLU should learn to extract.

The best way to get training data is
from *real users*, and a good way to get it is to
`pretend to be the bot yourself <https://medium.com/rasa-blog/put-on-your-robot-costume-and-be-the-minimum-viable-bot-yourself-3e48a5a59308>`_.
But to help get you started, we have some
`data saved <https://github.com/RasaHQ/rasa_nlu/blob/master/data/examples/rasa/demo-rasa.md>`_.


You can provide training data as json or markdown. 
We'll use markdown here because it's easier to read, but see :ref:`section_dataformat` for details.

Download the file (markdown format) and open it, and you'll see a list of
training examples, each composed of ``"text"``, ``"intent"`` and
``"entities"``, as shown below. In your working directory, create a
``data`` folder, and copy this ``demo-rasa.md`` file there.

.. code-block:: md

    ## intent:greet
    - hey
    - howdy
    - hey there
    ...


.. code-block:: md

    ## intent:restaurant_search
    - i'm looking for a place to eat
    - I want to grab lunch
    - I am searching for a dinner spot
    - i'm looking for a place in the [north](location) of town


Examples are grouped by intent, and entities are annotated as markdown links.
For details on the format see :ref:`section_dataformat`.

.. _training_your_model:

Training a New Model for your Project
-------------------------------------

Now we're going to create a configuration file. Make sure first that
you've set up a backend, see :ref:`section_backends`. Create a file
called ``config.yml`` in your working directory which looks like this
 
.. literalinclude:: ../sample_configs/config_spacy.yml
    :language: yaml

If you set up the tensorflow backend, you can use 

.. literalinclude:: ../sample_configs/config_embedding.yml
    :language: yaml


Now we can train your model by running:

.. code-block:: console

    $ python -m rasa_nlu.train \
        --config config.yml \
        --data data/examples/rasa/demo-rasa.json \
        --path projects

What do these parameters mean?

- **config**: configuration of the machine learning model
- **data**: file or folder that contains the training data. You can also
  pull training data from a URL using ``--url`` instead.
- **path**: path where the model will be saved


Full details of the parameters are in :ref:`section_pipeline`

After a few minutes, Rasa NLU will finish
training, and you'll see a new folder named 
``projects/default/model_YYYYMMDD-HHMMSS`` with the timestamp
when training finished.

.. _tutorial_using_your_model:

Using Your Model
----------------

By default, the server will look for all projects folders under the ``path``
directory specified. When no project is specified, as in this example,
a "default" one will be used, itself using the latest trained model.

.. code-block:: console

    $ python -m rasa_nlu.server --path projects

More information about starting the server can be found in :ref:`section_http`.

You can then test your new model by sending a request. Open a new window
on your terminal and run

.. note::

    **For windows users** the windows command line interface doesn't
    like single quotes. Use doublequotes and escape where necessary.
    ``curl -X POST "localhost:5000/parse" -d "{/"q/":/"I am looking for Mexican food/"}" | python -m json.tool``

.. code-block:: console

    $ curl -X POST localhost:5000/parse -d '{"q":"I am looking for Mexican food"}' | python -m json.tool

which should return 

.. code-block:: json

    {
        "intent": {
        "name": "restaurant_search",
        "confidence": 0.8231117999072759
        },
        "entities": [
            {
                "start": 17,
                "end": 24,
                "value": "mexican",
                "entity": "cuisine",
                "extractor": "ner_crf",
                "confidence": 0.875
            }
        ],
        "intent_ranking": [
            {
                "name": "restaurant_search",
                "confidence": 0.8231117999072759
            },
            {
                "name": "affirm",
                "confidence": 0.07618757211779097
            },
            {
                "name": "goodbye",
                "confidence": 0.06298664363805719
            },
            {
                "name": "greet",
                "confidence": 0.03771398433687609
            }
        ],
        "text": "I am looking for Mexican food"
    }

If not all of the entities aren't
found, don't panic! This tutorial is just a toy example, with far too
little training data to expect good performance.

.. note::

    Intent classification is independent of entity extraction, e.g.
    in "I am looking for Chinese food" the entities might not be extracted,
    though intent classification is correct.

Rasa NLU will also print a ``confidence`` value for the intent
classification. Note that the ``spacy_sklearn`` backend tends to report very low confidence scores. 
These are just a heuristic, not a true probability, and you shouldn't read too much into them.

You can use this to do some error handling in your chatbot (ex:
asking the user again if the confidence is low) and it's also
helpful for prioritising which intents need more training data.

.. note::
    The output may contain additional information, depending on the
    pipeline you are using. For example, not all pipelines include the
    ``"intent_ranking"`` information


With very little data, Rasa NLU can in certain cases
already generalise concepts, for example:

.. code-block:: console

    $ curl -X POST localhost:5000/parse -d '{"q":"I want some italian food"}' | python -m json.tool
    {
        "intent": {
            "name": "restaurant_search",
            "confidence": 0.5792111723774511
        },
        "entities": [
            {
                "entity": "cuisine",
                "value": "italian",
                "start": 12,
                "end": 19,
                "extractor": "ner_crf",
                "confidence": 0.875
            }
        ],
        "text": "I want some italian food"
    }

even though there's nothing quite like this sentence in
the examples used to train the model. To build a more robust app
you will obviously want to use a lot more training data, so go and collect it!

.. raw:: html 
   :file: poll.html
