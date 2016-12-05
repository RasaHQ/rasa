.. _section_tutorial:

.. _tutorial:

Tutorial: building a restaurant search bot
====================================

Note: see :ref:`section_migration` for how to clone your existing wit/LUIS/api.ai app.

As an example we'll use the domain of searching for restaurants. 
We'll start with an extremely simple model of those conversations. You can build up from there.

Let's assume that `anything` our bot's users say can be categorized into one of the following intents:

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

The first job of rasa NLU is to assign any given sentence to one of the categories: ``greet``, ``restaurant_search``, or ``thankyou``. 

The second job is to label words like "Mexican" and "center" as ``cuisine`` and ``location`` entities, respectively. 
In this tutorial we'll build a model which does exactly that.

Preparing the Training Data
------------------------------------

The best way to get training data is from *real users*, and the best way to do that is to `pretend to be the bot yourself <https://conversations.golastmile.com/put-on-your-robot-costume-and-be-the-minimum-viable-bot-yourself-3e48a5a59308#.d4tmdan68>`_. But to help get you started we have some data saved `here <https://github.com/golastmile/rasa_nlu/blob/master/data/demo-rasa.json>`_

Download the file and open it, and you'll see a list of training examples like these:


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

hopefully the format is intuitive if you've read this far into the tutorial.
In your working directory, create a ``data`` folder, and copy the ``demo-rasa.json`` file there.

It's always a good idea to `look` at your data before, during, and after training a model. 
To make this a bit simpler rasa NLU has a ``visualise`` tool, see :ref:`section_visualization`.
For the demo data the output should look like this:

.. image:: https://cloud.githubusercontent.com/assets/5114084/20884979/452df93c-bae6-11e6-8a2b-a6ad52306ae0.png


It is **strongly** recommended that you use the visualizer to do a sanity check before training.


Training Your Model
------------------------------------

Now we're going to create a configuration file. Make sure first that you've set up a backend, see :ref:`section_backends` .
Create a file called ``config.json`` in your working directory which looks like this

 
.. code-block:: json

    {
      "backend": "spacy_sklearn",
      "path" : "./",
      "data" : "./data/demo-restaurants.json"
    }

or if you've installed the MITIE backend instead:

 
.. code-block:: json

    {
      "backend": "mitie",
      "path" : "./",
      "mitie_file" : "path/to/total_word_feature_extractor.dat",
      "data" : "./data/demo-restaurants.json"
    }

Now we can train the model by running:

.. code-block:: console

    $ python -m rasa_nlu.train -c config.json

After a few minutes, rasa NLU will finish training, and you'll see a new dir called something like ``model_YYYYMMDD-HHMMSS`` with the timestamp when training finished. 

To run your trained model, add a ``server_model_dir`` to your ``config.json``: 

.. code-block:: json

    {
      "backend": "spacy_sklearn",
      "path" : "./",
      "data" : "./data/demo-restaurants.json",
      "server_model_dir" : "./model_YYYYMMDD-HHMMSS"
    }

and run the server with 


.. code-block:: console

    $ python -m rasa_nlu.server -c config.json

you can then test our your new model by sending a request. Open a new tab/window on your terminal and run


.. code-block:: console

    $ curl -XPOST localhost:5000/parse -d '{"text":"I am looking for Chinese food"}' | python -mjson.tool

which should return 

.. code-block:: json

    {
      "intent" : "restaurant_search",
      "entities" : [
        {
          "start": 8,
          "end": 15,
          "value": "chinese",
          "entity": "cuisine"
        }
      ]
    }

with very little data, rasa NLU can already generalise this concept, for example:


.. code-block:: console

    $ curl -XPOST localhost:5000/parse -d '{"text":"I want some italian"}' | python -mjson.tool
    {
      "entities": [
        {
          "end": 19,
          "entity": "cuisine",
          "start": 12,
          "value": "italian"
        }
      ],
      "intent": "restaurant_search",
      "text": "I want some italian"
    }

even though there's nothing quite like this sentence in the examples used to train the model. 
To build a more robust app you will obviously want to use a lot more data, so go and collect it!
