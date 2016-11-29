
Tutorial: building a restaurant search bot
====================================

As an example we'll use the domain of searching for restaurants. 
We'll start with an extremely simple model of those conversations, and build up from there.

Let's assume that `anything` our bot's users say can be categorized into one of the following intents:

- ``greet``
- ``search_restaurant``
- ``reject``
- ``thankyou``
- ``goodbye``

Of course there are many ways our users might ``greet`` our bot: [Hi! , Hey there!, Hello again].

There are also many ways to ``reject`` what our bot has suggested: [No, I don't like that place, My sister got sick eating there once].

The first job of rasa NLU is to assign any given sentence to one of these intents. 
For example, "Show me Mexican restaurants in the center of town" should map to ``search_restaurant``.

The second job is to extract "Mexican" and "center" as ``cuisine`` and ``location`` entities, respectively. 
In this tutorial we'll build a model which does exactly that. 

Training Data
------------------------------------

The best way to get training data is from *real users*, and the best way to do that is to `pretend to be the bot yourself <https://conversations.golastmile.com/put-on-your-robot-costume-and-be-the-minimum-viable-bot-yourself-3e48a5a59308#.d4tmdan68>`_. To help get you started we have some data saved `here <https://github.com/golastmile/rasa_nlu/blob/master/data/demo-rasa.json>`_

Download the file and open it, and you'll see a list of training examples like this one:


.. code-block:: json

    {
      "text": "hey", 
      "intent": "greet", 
      "entities": []
    }

.. code-block:: json

    {
      "text": "show me chinese restaurants", 
      "intent": "search_restaurant", 
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


Now we're going to create a configuration file. Make sure first that you've set up a backend, see :ref:`section_migration` .
Create a file called ``config.json`` which looks like this

 
.. code-block:: json

    {
      "backend": "spacy_sklearn",
    }

