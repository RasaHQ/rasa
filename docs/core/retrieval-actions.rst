:desc: Use a retrieval model to select chatbot responses a
       in open source bot framework Rasa. a
 a
.. _retrieval-actions: a
 a
Retrieval Actions a
================= a
 a
.. edit-link:: a
 a
.. warning:: a
   This feature is experimental. a
   We introduce experimental features to get feedback from our community, so we encourage you to try it out! a
   However, the functionality might be changed or removed in the future. a
   If you have feedback (positive or negative) please share it with us on the `forum <https://forum.rasa.com>`_. a
   Also, currently we do not support adding new annotations in Rasa X if your training data contains retrieval actions. a
   Once we have gathered enough feedback and we're happy with the training data format, we'll add support for training response retrieval models in Rasa X. a
 a
.. note:: a
   There is an in-depth blog post `here <https://blog.rasa.com/response-retrieval-models/>`_ about how to use retrieval a
   actions for handling single turn interactions. a
 a
.. contents:: a
   :local: a
 a
About a
^^^^^ a
 a
Retrieval actions are designed to make it simpler to work with :ref:`small-talk` and :ref:`simple-questions` . a
For example, if your assistant can handle 100 FAQs and 50 different small talk intents, you can use a single retrieval a
action to cover all of these. a
From a dialogue perspective, these single-turn exchanges can all be treated equally, so this simplifies your stories. a
 a
Instead of having a lot of stories like: a
 a
.. code-block:: story a
 a
   ## weather a
   * ask_weather a
      - utter_ask_weather a
    a
   ## introduction a
   * ask_name a
      - utter_introduce_myself a
 a
   ... a
 a
 a
You can cover all of these with a single story where the above intents are grouped under a common ``chitchat`` intent: a
 a
 a
.. code-block:: story a
 a
   ## chitchat a
   * chitchat a
      - respond_chitchat a
 a
A retrieval action uses the output of a :ref:`response-selector` component from NLU which learns a a
retrieval model to predict the correct response from a list of candidate responses given a user message text. a
 a
 a
.. _retrieval-training-data: a
 a
Training Data a
^^^^^^^^^^^^^ a
 a
Like the name suggests, retrieval actions learn to select the correct response from a list of candidates. a
As with other NLU data, you need to include examples of what your users will say in your NLU file: a
 a
.. code-block:: md a
 a
   ## intent: chitchat/ask_name a
   - what's your name a
   - who are you? a
   - what are you called? a
 a
   ## intent: chitchat/ask_weather a
   - how's weather? a
   - is it sunny where you are? a
 a
First, all of these examples will be combined into a single ``chitchat`` retrieval intent that NLU will predict. a
All retrieval intents have a suffix added to them which identifies a particular response text for your assistant, in the a
above example - ``ask_name`` and ``ask_weather``. The suffix is separated from the intent name by a ``/`` delimiter a
 a
Next, include response texts for all retrieval intents in a **separate** training data file as ``responses.md``: a
 a
.. code-block:: md a
 a
    ## ask name a
    * chitchat/ask_name a
        - my name is Sara, Rasa's documentation bot! a
 a
    ## ask weather a
    * chitchat/ask_weather a
        - it's always sunny where I live a
 a
The retrieval model is trained separately as part of the NLU training pipeline to select the correct response. a
One important thing to remember is that the retrieval model uses the text of the response messages a
to select the correct one. If you change the text of these responses, you have to retrain your retrieval model! a
This is a key difference to the responses defined in your domain file. a
 a
.. note:: a
    The file containing response texts must exist as a separate file inside the training data directory passed a
    to the training process. The contents of it cannot be a part of the file which contains training data for other a
    components of NLU. a
 a
.. note:: a
    As shown in the above examples, ``/`` symbol is reserved as a delimiter to separate retrieval intents from response text identifier. Make sure not to a
    use it in the name of your intents. a
 a
Config File a
^^^^^^^^^^^ a
 a
You need to include the :ref:`response-selector` component in your config. The component needs a tokenizer, a featurizer and an a
intent classifier to operate on the user message before it can predict a response and hence these a
components should be placed before ``ResponseSelector`` in the NLU configuration. An example: a
 a
.. code-block:: yaml a
 a
    language: "en" a
 a
    pipeline: a
    - name: "WhitespaceTokenizer" a
      intent_split_symbol: "_" a
    - name: "CountVectorsFeaturizer" a
    - name: "DIETClassifier" a
    - name: "ResponseSelector" a
 a
Domain a
^^^^^^ a
 a
Rasa uses a naming convention to match the intent names like ``chitchat/ask_name`` a
to the retrieval action.  a
The correct action name in this case is ``respond_chitchat``. The prefix ``respond_`` is mandatory to identify it as a a
retrieval action. Another example - correct action name for ``faq/ask_policy`` would be ``respond_faq`` a
To include this in your domain, add it to the list of actions: a
 a
.. code-block:: yaml a
 a
   actions: a
     ... a
     - respond_chitchat a
     - respond_faq a
 a
 a
A simple way to ensure that the retrieval action is predicted after the chitchat a
intent is to use the :ref:`mapping-policy`. a
However, you can also include this action in your stories. a
For example, if you want to repeat a question after handling chitchat a
(see :ref:`unhappy-paths` ) a
 a
.. code-block:: story a
 a
   ## interruption a
   * search_restaurant a
      - utter_ask_cuisine a
   * chitchat a
      - respond_chitchat a
      - utter_ask_cuisine a
 a
Multiple Retrieval Actions a
^^^^^^^^^^^^^^^^^^^^^^^^^^ a
 a
If your assistant includes both FAQs **and** chitchat, it is possible to a
separate these into separate retrieval actions, for example having intents a
like ``chitchat/ask_weather`` and ``faq/returns_policy``. a
Rasa supports adding multiple ``RetrievalActions`` like ``respond_chitchat`` and ``respond_returns_policy`` a
To train separate retrieval models for each of the intents, you need to include a separate ``ResponseSelector`` a
component in the config: a
 a
.. code-block:: yaml a
 a
    language: "en" a
 a
    pipeline: a
    - name: "WhitespaceTokenizer" a
      intent_split_symbol: "_" a
    - name: "CountVectorsFeaturizer" a
    - name: "DIETClassifier" a
    - name: "ResponseSelector" a
      retrieval_intent: chitchat a
    - name: "ResponseSelector" a
      retrieval_intent: faq a
 a
You could still have two separate retrieval actions but both actions can share the same retrieval model by specifying a single a
 ``ResponseSelector`` component and leaving the ``retrieval_intent`` to its default value(None): a
 a
.. code-block:: yaml a
 a
    language: "en" a
 a
    pipeline: a
    - name: "WhitespaceTokenizer" a
      intent_split_symbol: "_" a
    - name: "CountVectorsFeaturizer" a
    - name: "DIETClassifier" a
    - name: "ResponseSelector" a
 a
 a
In this case, the response selector will be trained on examples from both ``chitchat/{x}`` and ``faq/{x}`` and will be a
identified by the name ``default`` the NLU parsed output. a
 a
In our experiments so far, having separate retrieval models does **not** make any difference to the accuracy of each a
retrieval action. So for simplicity, we recommend you use a single retrieval a
model for both chitchat and FAQs a
If you get different results, please let us know in the `forum <https://forum.rasa.com>`_ ! a
 a
 a
Parsing Response Selector Output a
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ a
 a
The parsed output from NLU will have a property named ``response_selector`` containing the output for a
each response selector. Each response selector is identified by ``retrieval_intent`` parameter of that response selector a
and stores two properties - a
 a
    - ``response``: The predicted response text and the prediction confidence. a
    - ``ranking``: Ranking with confidences of top 10 candidate responses. a
 a
Example result: a
 a
.. code-block:: json a
 a
    { a
        "text": "What is the recommend python version to install?", a
        "entities": [], a
        "intent": {"confidence": 0.6485910906220309, "name": "faq"}, a
        "intent_ranking": [ a
            {"confidence": 0.6485910906220309, "name": "faq"}, a
            {"confidence": 0.1416153159565678, "name": "greet"} a
        ], a
        "response_selector": { a
          "faq": { a
            "response": {"confidence": 0.7356462617, "name": "Supports 3.5, 3.6 and 3.7, recommended version is 3.6"}, a
            "ranking": [ a
                {"confidence": 0.7356462617, "name": "Supports 3.5, 3.6 and 3.7, recommended version is 3.6"}, a
                {"confidence": 0.2134543431, "name": "You can ask me about how to get started"} a
            ] a
          } a
        } a
    } a
 a
If the ``retrieval_intent`` parameter of a particular response selector was left to its default value, a
the corresponding response selector will be identified as ``default`` in the returned output. a
 a