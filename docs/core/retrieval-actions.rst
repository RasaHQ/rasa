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

.. contents::
   :local:

About
^^^^^

Retrieval actions are designed to make it simpler to work with :ref:`small-talk` and :ref:`simple-questions` .
For example, if your assistant can handle 100 FAQs and 50 different small talk intents,
You can use a single retrieval action to cover all of these. 
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


You can cover all of these with a single story:


.. code-block:: story

   ## chitchat
   * chitchat
      - respond_chitchat


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

First, all of these examples will be combined into a single ``chitchat`` intent that NLU will predict.

The retrieval model is trained separately to select the correct response. 
TODO: explain where this file needs to go.

.. code-block:: md

    * chitchat/ask_name
        - my name is Sara, Rasa's documentation bot!

    * chitchat/ask_weather
        - it's always sunny where I live

In important thing to remember is that the retrieval model uses the text of the response messages
to select the correct one.
If you change the text of these responses, you have to retrain your retrieval model!
This is a key difference to the response templates in your domain file. 

Config File
^^^^^^^^^^^

You need to include the ``ResponseSelector`` component in your config.

Domain
^^^^^^

Rasa uses a naming convention to match the intent names like ``chitchat/ask_name``
to the retrieval action. 
The correct action name in this case is ``respond_chitchat``.
To include this in your domain, add it to the list of actions:

.. code-block:: yaml

   actions:
     ...
     - respond_chitchat


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
Rasa supports adding multiple ``RetrievalActions``. If you do this, a separate
retrieval model will be trained for the ``chitchat/{x}`` and ``faq/{x}`` intents.
In our experiments so far, this does **not** make any difference to the accuracy
of the retrieval models. So for simplicity, we recommend you use a single retrieval
action for both chitchat and FAQs. If you get different results, please let us know 
in the :ref:`forum <https://forum.rasa.com>` !