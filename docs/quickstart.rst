:desc: Try out Rasa NLU in the browser.

.. _section_quickstart:

.. _tutorial:

Getting Started with Rasa NLU
=============================

In this tutorial you will create your first Rasa NLU bot. You can run all of the 
code snippets in here directly, or you can install Rasa NLU and run the examples on your
own machine.


As an example we'll start a new project to help people search for restaurants. 
We'll start with an extremely simple model of those conversations. You can build up from there.

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

1. Prepare your NLU Training Data
---------------------------------

Training data is essential for developing chatbots and voice apps. 
The data is just a list of messages that you expect to receive, annotated with 
the intent and entities Rasa NLU should learn to extract.

The best way to get training data is from *real users*, and a good way to get it is to
`pretend to be the bot yourself <https://medium.com/rasa-blog/put-on-your-robot-costume-and-be-the-minimum-viable-bot-yourself-3e48a5a59308>`_.
But to help get you started, we have some demo data here.
See :ref:`section_dataformat` for details of the data format.

If you are running this in the docs, it may take a few seconds to start up.
If you are running locally, copy the text between the triple quotes (``"""``)
and save it in a file called ``nlu.md``.

.. runnable::
   :description: nlu-write-nlu-data

   nlu_md = """
   ## intent:greet
   - hey
   - hello
   - hi
   - good morning
   - good evening
   - hey there

   ## intent:restaurant_search
   - i'm looking for a place to eat
   - I want to grab lunch
   - I am searching for a dinner spot
   - i'm looking for a place in the [north](location) of town
   - show me [chinese](cuisine) restaurants
   - show me a [mexican](cuisine) place in the [centre](location)
   - i am looking for an [indian](cuisine) spot
   - search for restaurants
   - anywhere in the [west](location)
   - anywhere near [18328](location)
   - I am looking for [asian fusion](cuisine) food
   - I am looking a restaurant in [29432](location)

   ## intent:thankyou
   - thanks! 
   - thank you
   - thx
   - thanks very much
   """
   %store nlu_md > nlu.md


2. Define your Machine Learning Model
-------------------------------------

Rasa NLU has a number of different components, which together make a pipeline. Create a markdown file with the pipeline you want to use. In this case, we're using the pre-defined ``tensorflow_embedding`` pipeline. If you are running this locally instead of here in the docs, copy the text between the (``"""``)
and save it in a file called ``nlu_config.yml``.

.. runnable:: 
   :description: nlu-write-nlu-config

   nlu_config = """
   language: en
   pipeline: tensorflow_embedding
   """
   %store nlu_config > nlu_config.yml


Full details of the pipeline components are in :ref:`section_pipeline`


3. Train your Machine Learning NLU model.
-----------------------------------------

To train a model, start the ``rasa_nlu.train`` command, and tell it where to find your configuration and your training data:

If you are running this in your computer, leave out the ``!`` at the start.

.. runnable::
   :description: nlu-train-nlu

   !python -m rasa_nlu.train -c nlu_config.yml --data nlu.md -o models --fixed_model_name nlu --project current --verbose


We are also passing the ``--project current`` and ``--fixed_model_name nlu`` parameters, this means the model will be saved at ``.models/current/nlu`` relative to your working directory.


.. _tutorial_using_your_model:

4. Try it out!
--------------

There are two ways you can use your model, directly from python, or by starting a http server. 

To use your new model in python, create an ``Interpreter`` object and pass a message to its ``parse()`` method.

.. runnable::
    :description: nlu-parse-nlu-python

    from rasa_nlu.model import Interpreter
    import json
    interpreter = Interpreter.load("./models/current/nlu")
    message = "let's see some italian restaurants"
    result = interpreter.parse(message)
    print(json.dumps(result, indent=2))


5. Start your Rasa NLU HTTP Server
----------------------------------


Run this command to start your server:

.. runnable::
   :description: nlu-start-server

   !python -m rasa_nlu.server --path models


By default, the server will look for all projects folders under the ``path``

Let's try it out with a HTTP request:

.. runnable::
   :description: nlu-curl-server

   !curl 'localhost:5000/parse?q=hello&project=current&model=nlu' | python -m json.tool


More information about starting the server can be found in :ref:`section_configuration`.

6. Start Building
-----------------

Clone the starter pack 

.. copyable::

   git clone https://github.com/RasaHQ/starter-pack.git

The starter pack gets you set up with the right file structure, sample configurations, plus
links to more training data! 

Bonus Material
--------------

With very little data, Rasa NLU can in certain cases
already generalise concepts, for example:

.. runnable:: 
    :description: nlu-parse-2

    from rasa_nlu.model import Interpreter
    import json

    interpreter = Interpreter.load("./models/current/nlu")
    message = "I want some italian food"
    result = interpreter.parse(message)
    print(json.dumps(message), indent=2)


even though there's nothing quite like this sentence in
the examples used to train the model. To build a more robust app
you will obviously want to use a lot more training data, so go and collect it!


.. note::

    **For windows users** the windows command line interface doesn't
    like single quotes. Use doublequotes and escape where necessary.
    ``curl -X POST "localhost:5000/parse" -d "{/"q/":/"I am looking for Mexican food/"}" | python -m json.tool``

Spend some time playing around with the commands above, sending some different test messages to Rasa NLU. 
Remember that this is just a toy example, with just a little bit of training data. 
To build a really great NLU system you'll want to collect a lot more real user messages.

.. note::

    Intent classification is independent of entity extraction. So sometimes
    NLU will get the intent right but entities wrong, or the other way around. 
    You need to provide enough data for both intents and entities. 

Rasa NLU will also print a ``confidence`` value for the intent
classification. Note that the ``spacy_sklearn`` backend tends to report very low confidence scores. 
These are just a heuristic, not a true probability, and you shouldn't read too much into them. Read :ref:`section_fallback` for more details.


.. note::
    The output may contain additional information, depending on the
    pipeline you are using. For example, not all pipelines include the
    ``"intent_ranking"`` information


.. raw:: html 
   :file: poll.html
