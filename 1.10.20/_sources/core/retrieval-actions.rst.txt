:desc: Use a retrieval model to select chatbot responses
       in open source bot framework Rasa.

.. _retrieval-actions:

Retrieval Actions
=================

.. edit-link::

.. warning::
   This feature is experimental.
   We introduce experimental features to get feedback from our community, so we encourage you to try it out!
   However, the functionality might be changed or removed in the future.
   If you have feedback (positive or negative) please share it with us on the `forum <https://forum.rasa.com>`_.
   Also, currently we do not support adding new annotations in Rasa X if your training data contains retrieval actions.
   Once we have gathered enough feedback and we're happy with the training data format, we'll add support for training response retrieval models in Rasa X.

.. note::
   There is an in-depth blog post `here <https://blog.rasa.com/response-retrieval-models/>`_ about how to use retrieval
   actions for handling single turn interactions.

.. contents::
   :local:

About
^^^^^

Retrieval actions are designed to make it simpler to work with :ref:`small-talk` and :ref:`simple-questions` .
For example, if your assistant can handle 100 FAQs and 50 different small talk intents, you can use a single retrieval
action to cover all of these.
From a dialogue perspective, these single-turn exchanges can all be treated equally, so this simplifies your stories.

Instead of having a lot of stories like:

.. code-block:: story

   ## weather
   * ask_weather
      - utter_ask_weather
   
   ## introduction
   * ask_name
      - utter_introduce_myself

   ...


You can cover all of these with a single story where the above intents are grouped under a common ``chitchat`` intent:


.. code-block:: story

   ## chitchat
   * chitchat
      - respond_chitchat

A retrieval action uses the output of a :ref:`response-selector` component from NLU which learns a
retrieval model to predict the correct response from a list of candidate responses given a user message text.


.. _retrieval-training-data:

Training Data
^^^^^^^^^^^^^

Like the name suggests, retrieval actions learn to select the correct response from a list of candidates.
As with other NLU data, you need to include examples of what your users will say in your NLU file:

.. code-block:: md

   ## intent: chitchat/ask_name
   - what's your name
   - who are you?
   - what are you called?

   ## intent: chitchat/ask_weather
   - how's weather?
   - is it sunny where you are?

First, all of these examples will be combined into a single ``chitchat`` retrieval intent that NLU will predict.
All retrieval intents have a suffix added to them which identifies a particular response text for your assistant, in the
above example - ``ask_name`` and ``ask_weather``. The suffix is separated from the intent name by a ``/`` delimiter

Next, include response texts for all retrieval intents in a **separate** training data file as ``responses.md``:

.. code-block:: md

    ## ask name
    * chitchat/ask_name
        - my name is Sara, Rasa's documentation bot!

    ## ask weather
    * chitchat/ask_weather
        - it's always sunny where I live

The retrieval model is trained separately as part of the NLU training pipeline to select the correct response.
One important thing to remember is that the retrieval model uses the text of the response messages
to select the correct one. If you change the text of these responses, you have to retrain your retrieval model!
This is a key difference to the response templates in your domain file.

.. note::
    The file containing response texts must exist as a separate file inside the training data directory passed
    to the training process. The contents of it cannot be a part of the file which contains training data for other
    components of NLU.

.. note::
    As shown in the above examples, ``/`` symbol is reserved as a delimiter to separate retrieval intents from response text identifier. Make sure not to
    use it in the name of your intents.

Config File
^^^^^^^^^^^

You need to include the :ref:`response-selector` component in your config. The component needs a tokenizer, a featurizer and an
intent classifier to operate on the user message before it can predict a response and hence these
components should be placed before ``ResponseSelector`` in the NLU configuration. An example:

.. code-block:: yaml

    language: "en"

    pipeline:
    - name: "WhitespaceTokenizer"
      intent_split_symbol: "_"
    - name: "CountVectorsFeaturizer"
    - name: "EmbeddingIntentClassifier"
    - name: "ResponseSelector"

Domain
^^^^^^

Rasa uses a naming convention to match the intent names like ``chitchat/ask_name``
to the retrieval action. 
The correct action name in this case is ``respond_chitchat``. The prefix ``respond_`` is mandatory to identify it as a
retrieval action. Another example - correct action name for ``faq/ask_policy`` would be ``respond_faq``
To include this in your domain, add it to the list of actions:

.. code-block:: yaml

   actions:
     ...
     - respond_chitchat
     - respond_faq


A simple way to ensure that the retrieval action is predicted after the chitchat
intent is to use the :ref:`mapping-policy`.
However, you can also include this action in your stories.
For example, if you want to repeat a question after handling chitchat
(see :ref:`unhappy-paths` )

.. code-block:: story

   ## interruption
   * search_restaurant
      - utter_ask_cuisine
   * chitchat
      - respond_chitchat
      - utter_ask_cuisine

Multiple Retrieval Actions
^^^^^^^^^^^^^^^^^^^^^^^^^^

If your assistant includes both FAQs **and** chitchat, it is possible to
separate these into separate retrieval actions, for example having intents
like ``chitchat/ask_weather`` and ``faq/returns_policy``.
Rasa supports adding multiple ``RetrievalActions`` like ``respond_chitchat`` and ``respond_returns_policy``
To train separate retrieval models for each of the intents, you need to include a separate ``ResponseSelector``
component in the config:

.. code-block:: yaml

    language: "en"

    pipeline:
    - name: "WhitespaceTokenizer"
      intent_split_symbol: "_"
    - name: "CountVectorsFeaturizer"
    - name: "EmbeddingIntentClassifier"
    - name: "ResponseSelector"
      retrieval_intent: chitchat
    - name: "ResponseSelector"
      retrieval_intent: faq

You could still have two separate retrieval actions but both actions can share the same retrieval model by specifying a single
 ``ResponseSelector`` component and leaving the ``retrieval_intent`` to its default value(None):

.. code-block:: yaml

    language: "en"

    pipeline:
    - name: "WhitespaceTokenizer"
      intent_split_symbol: "_"
    - name: "CountVectorsFeaturizer"
    - name: "EmbeddingIntentClassifier"
    - name: "ResponseSelector"


In this case, the response selector will be trained on examples from both ``chitchat/{x}`` and ``faq/{x}`` and will be
identified by the name ``default`` the NLU parsed output.

In our experiments so far, having separate retrieval models does **not** make any difference to the accuracy of each
retrieval action. So for simplicity, we recommend you use a single retrieval
model for both chitchat and FAQs
If you get different results, please let us know in the `forum <https://forum.rasa.com>`_ !


Parsing Response Selector Output
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The parsed output from NLU will have a property named ``response_selector`` containing the output for
each response selector. Each response selector is identified by ``retrieval_intent`` parameter of that response selector
and stores two properties -

    - ``response``: The predicted response text and the prediction confidence.
    - ``ranking``: Ranking with confidences of top 10 candidate responses.

Example result:

.. code-block:: json

    {
        "text": "What is the recommend python version to install?",
        "entities": [],
        "intent": {"confidence": 0.6485910906220309, "name": "faq"},
        "intent_ranking": [
            {"confidence": 0.6485910906220309, "name": "faq"},
            {"confidence": 0.1416153159565678, "name": "greet"}
        ],
        "response_selector": {
          "faq": {
            "response": {"confidence": 0.7356462617, "name": "Supports 3.5, 3.6 and 3.7, recommended version is 3.6"},
            "ranking": [
                {"confidence": 0.7356462617, "name": "Supports 3.5, 3.6 and 3.7, recommended version is 3.6"},
                {"confidence": 0.2134543431, "name": "You can ask me about how to get started"}
            ]
          }
        }
    }

If the ``retrieval_intent`` parameter of a particular response selector was left to its default value,
the corresponding response selector will be identified as ``default`` in the returned output.
