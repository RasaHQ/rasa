:desc: How to build simple FAQ and contextual assistants

.. _tutorial-faq-assistants:

Building FAQ Assistants
=======================

.. edit-link::

After following the basics of setting up an assistant in the `Rasa Tutorial <https://rasa.com/docs/rasa/user-guide/rasa-tutorial/>`_, we'll
now walk through building a basic FAQ assistant.

.. contents::
   :local:

.. _build-faq-assistant:

Building a simple FAQ assistant
-------------------------------

FAQ assistants are the simplest assistants to build and a good place to get started.
These assistants allow the user to ask a simple question and get a response. We’re going to
build a basic FAQ assistant using features of Rasa designed specifically for this type of assistant.

In this section we’re going to cover the following topics:

    - `Responding to simple intents <respond-with-memoization-policy>`_ with the MemoizationPolicy
    - `Handling FAQs <faqs-response-selector>`_ using the ResponseSelector

.. note::
    Here's a minimal checklist of what you need to build a basic FAQ assistant:

      - ``data/nlu.md``: NLU training data for ``faq/`` intents
      - ``data/responses.md``: responses associated with ``faq/`` intents
      - ``config.yml``: ``ReponseSelector`` in your NLU pipeline
      - ``domain.yml``: a retrieval action ``respond_faq`` and intent ``faq``
      - ``data/stories.md``: a simple story for FAQs
      - ``test_stories.md``: E2E test stories for your FAQs


We’re going to use content from `Sara <https://github.com/RasaHQ/rasa-demo>`_, the Rasa
assistant that, amongst other things, helps the user get started with the Rasa products.
You should first install Rasa using the `Step-by-step Installation Guide <https://rasa.com/docs/rasa/user-guide/installation/#step-by-step-installation-guide>`_
and then follow the `Rasa Tutorial <https://rasa.com/docs/rasa/user-guide/rasa-tutorial/>`_
to make sure you know the basics.

To prepare for this tutorial, we're going to create a new directory and start a
new Rasa project.

.. code-block:: bash

    mkdir rasa-assistant
    rasa init


Let's remove the default content from this bot, so that the ``nlu.md``, ``stories.md``
and ``domain.yml`` files are empty.

.. _respond-with-memoization-policy:

Memoization Policy
^^^^^^^^^^^^^^^^^^

The MemoizationPolicy remembers examples from training stories for up to a ``max_history``
of turns. The number of "turns" includes messages the user sent, and actions the
assistant performed. For the purpose of a simple, context-less FAQ bot, we only need
to pay attention to the last message the user sent, and therefore we’ll set that to ``1``.

You can do this by editing your ``config.yml`` file as follows (you can remove ``TEDPolicy`` for now):

.. code-block:: yaml

  policies:
  - name: MemoizationPolicy
    max_history: 1
  - name: MappingPolicy

.. note::
   The MappingPolicy is there because it handles the logic of the ``/restart`` intent,
   which allows you to clear the conversation history and start fresh.

Now that we’ve defined our policies, we can add some stories for the ``goodbye``, ``thank`` and ``greet``
intents to the ``stories.md`` file:

.. code-block:: md

   ## greet
   * greet
     - utter_greet

   ## thank
   * thank
     - utter_noworries

   ## goodbye
   * bye
     - utter_bye

We’ll also need to add the intents, actions and responses to our ``domain.yml`` file in the following sections:

.. code-block:: md

   intents:
     - greet
     - bye
     - thank

   responses:
     utter_noworries:
       - text: No worries!
     utter_greet:
       - text: Hi
     utter_bye:
       - text: Bye!

Finally, we’ll copy over some NLU data from Sara into our ``nlu.md`` file
(more can be found `here <https://github.com/RasaHQ/rasa-demo/blob/master/data/nlu/nlu.md>`__):

.. code-block:: md

   ## intent:greet
   - Hi
   - Hey
   - Hi bot
   - Hey bot
   - Hello
   - Good morning
   - hi again
   - hi folks

   ## intent:bye
   - goodbye
   - goodnight
   - good bye
   - good night
   - see ya
   - toodle-oo
   - bye bye
   - gotta go
   - farewell

   ## intent:thank
   - Thanks
   - Thank you
   - Thank you so much
   - Thanks bot
   - Thanks for that
   - cheers

You can now train a first model and test the bot, by running the following commands:

.. code-block:: bash

   rasa train
   rasa shell

This bot should now be able to reply to the intents we defined consistently, and in any order.

For example, it should be able to do:

.. code-block:: bash

    Bot loaded. Type a message and press enter (use '/stop' to exit): 
    Your input ->  bye!
    Bye!
    Your input ->  thanks!
    No worries!
    Your input ->  hi!
    Hi
    Your input ->  goodbye
    Bye!
    Your input ->  thank you
    No worries!


While it's good to test the bot interactively, we should also add end to end test cases that
can later be included as part of our CI/CD system. `End to end stories <https://rasa.com/docs/rasa/user-guide/evaluating-models/#end-to-end-evaluation>`_
include NLU data, so that both components of Rasa can be tested.  Create a file called
``test_stories.md`` in the root directory with some test cases:

.. code-block:: md

   ## greet + goodbye
   * greet: Hi!
     - utter_greet
   * bye: Bye
     - utter_bye

   ## greet + thanks
   * greet: Hello there
     - utter_greet
   * thank: thanks a bunch
     - utter_noworries

   ## greet + thanks + goodbye
   * greet: Hey
     - utter_greet
   * thank: thank you
     - utter_noworries
   * bye: bye bye
     - utter_bye

To test our model against the test file, run the command:

.. code-block:: bash

   rasa test --e2e --stories test_stories.md

The test command will produce a directory named ``results``. It should contain a file
called ``failed_stories.md``, where any test cases that failed will be printed. It will
also specify whether it was an NLU or Core prediction that went wrong.  As part of a
CI/CD pipeline, the test option ``--fail-on-prediction-errors`` can be used to throw
an exception that stops the pipeline.

.. _faqs-response-selector:

Response Selectors
^^^^^^^^^^^^^^^^^^

The :ref:`response-selector` NLU component is designed to make it easier to handle dialogue
elements like :ref:`small-talk` and FAQ messages in a simple manner. By using the ResponseSelector,
you only need one story to handle all FAQs, instead of adding new stories every time you
want to increase your bot's scope.

People often ask Sara different questions surrounding the Rasa products, so let’s
start with three intents: ``ask_channels``, ``ask_languages``, and ``ask_rasax``.
We’re going to copy over some NLU data from the `Sara training data <https://github.com/RasaHQ/rasa-demo/blob/master/data/nlu/nlu.md>`_
into our ``nlu.md``. It’s important that these intents have an ``faq/`` prefix, so they’re
recognised as the faq intent by the ResponseSelector:

.. code-block:: md

   ## intent: faq/ask_channels
   - What channels of communication does rasa support?
   - what channels do you support?
   - what chat channels does rasa uses
   - channels supported by Rasa
   - which messaging channels does rasa support?

   ## intent: faq/ask_languages
   - what language does rasa support?
   - which language do you support?
   - which languages supports rasa
   - can I use rasa also for another laguage?
   - languages supported

   ## intent: faq/ask_rasax
   - I want information about rasa x
   - i want to learn more about Rasa X
   - what is rasa x?
   - Can you tell me about rasa x?
   - Tell me about rasa x
   - tell me what is rasa x

Next, we’ll need to define the responses associated with these FAQs in a new file called ``responses.md`` in the ``data/`` directory:

.. code-block:: md

   ## ask channels
   * faq/ask_channels
     - We have a comprehensive list of [supported connectors](https://rasa.com/docs/core/connectors/), but if
       you don't see the one you're looking for, you can always create a custom connector by following
       [this guide](https://rasa.com/docs/rasa/user-guide/connectors/custom-connectors/).

   ## ask languages
   * faq/ask_languages
     - You can use Rasa to build assistants in any language you want!

   ## ask rasa x
   * faq/ask_rasax
    - Rasa X is a tool to learn from real conversations and improve your assistant. Read more [here](https://rasa.com/docs/rasa-x/)

The ResponseSelector should already be at the end of the NLU pipeline in our ``config.yml``:

.. code-block:: yaml

    language: en
    pipeline:
      - name: WhitespaceTokenizer
      - name: RegexFeaturizer
      - name: LexicalSyntacticFeaturizer
      - name: CountVectorsFeaturizer
      - name: CountVectorsFeaturizer
        analyzer: "char_wb"
        min_ngram: 1
        max_ngram: 4
      - name: DIETClassifier
        epochs: 100
      - name: EntitySynonymMapper
      - name: ResponseSelector
        epochs: 100

Now that we’ve defined the NLU side, we need to make Core aware of these changes. Open your ``domain.yml`` file and add the ``faq`` intent:

.. code-block:: yaml

   intents:
     - greet
     - bye
     - thank
     - faq

We’ll also need to add a `retrieval action <https://rasa.com/docs/rasa/core/retrieval-actions/>`_,
which takes care of sending the response predicted from the  ResponseSelector back to the user,
to the list of actions. These actions always have to start with the ``respond_`` prefix:

.. code-block:: yaml

   actions:
     - respond_faq

Next we’ll write a story so that Core knows which action to predict:

.. code-block:: md

   ## Some question from FAQ
   * faq
       - respond_faq

This prediction is handled by the MemoizationPolicy, as we described earlier.

After all of the changes are done, train a new model and test the modified FAQs:

.. code-block:: bash

   rasa train
   rasa shell

At this stage it makes sense to add a few test cases to your ``test_stories.md`` file again:

.. code-block:: md

   ## ask channels
   * faq: What messaging channels does Rasa support?
     - respond_faq

   ## ask languages
   * faq: Which languages can I build assistants in?
     - respond_faq

   ## ask rasa x
   * faq: What’s Rasa X?
     - respond_faq

You can read more in this `blog post <https://blog.rasa.com/response-retrieval-models/>`_ and the
`Retrieval Actions <https://rasa.com/docs/rasa/core/retrieval-actions/>`_ page.

Using the features we described in this tutorial, you can easily build a context-less assistant.
When you’re ready to enhance your assistant with context, check out :ref:`tutorial-contextual-assistants`.
