:desc: Define custom fallback actions with thresholds for NLU and Core for letting
       your conversation fail gracefully with open source dialogue management.

.. _fallbacks:

Fallback Actions
================

Sometimes you want to revert to a fallback action, such as replying,
`"Sorry, I didn't understand that"`. You can handle fallback cases by adding
either the ``FallbackPolicy`` or the ``TwoStageFallbackPolicy`` to your
policy ensemble.

Fallback Policy
---------------


The ``FallbackPolicy`` has one fallback action, which will
be executed if the intent recognition has a confidence below ``nlu_threshold``
or if none of the dialogue policies predict an action with
confidence higher than ``core_threshold``.

The thresholds and fallback action can be adjusted in the policy configuration
file as parameters of the ``FallbackPolicy``.

.. code-block:: yaml

  policies:
    - name: "FallbackPolicy"
      nlu_threshold: 0.4
      core_threshold: 0.3
      fallback_action_name: "action_default_fallback"

``action_default_fallback`` is a default action in Rasa Core which sends the
``utter_default`` template message to the user. Make sure to specify
the ``utter_default`` in your domain file. It will also revert back to the
state of the conversation before the user message that caused the
fallback, so that it will not influence the prediction of future actions.
You can take a look at the source of the action below:

.. autoclass:: rasa.core.actions.action.ActionDefaultFallback


You can also create your own custom action to use as a fallback (see
:ref:`customactions` for more info on custom actions). If you
do, make sure to pass the custom fallback action to ``FallbackPolicy`` inside
your policy configuration file. For example:

.. code-block:: yaml

  policies:
    - name: "FallbackPolicy"
      nlu_threshold: 0.4
      core_threshold: 0.3
      fallback_action_name: "my_fallback_action"


.. note::
  If your custom fallback action does not return a ``UserUtteranceReverted`` event,
  the next predictions of your bot may become inaccurate, as it is very likely that
  the fallback action is not present in your stories.

If you have a specific intent, let's say it's called ``out_of_scope``, that
should always trigger the fallback action, you should add this as a story:

.. code-block:: story

    ## fallback story
    * out_of_scope
      - action_default_fallback


Two-stage Fallback Policy
-------------------------

The ``TwoStageFallbackPolicy`` handles low NLU confidence in multiple stages
by trying to disambiguate the user input (low core confidence is handled in
the same manner as the ``FallbackPolicy``).

- If a NLU prediction has a low confidence score, the user is asked to affirm
  the classification of the intent.  (Default action:
  ``action_default_ask_affirmation``)

    - If they affirm, the story continues as if the intent was classified
      with high confidence from the beginning.
    - If they deny, the user is asked to rephrase their message.

- Rephrasing  (default action: ``action_default_ask_rephrase``)

    - If the classification of the rephrased intent was confident, the story
      continues as if the user had this intent from the beginning.
    - If the rephrased intent was not classified with high confidence, the user
      is asked to affirm the classified intent.

- Second affirmation  (default action: ``action_default_ask_affirmation``)

    - If the user affirms the intent, the story continues as if the user had
      this intent from the beginning.
    - If the user denies, the original intent is classified as the specified
      ``deny_suggestion_intent_name``, and an ultimate fallback action
      ``fallback_nlu_action_name`` is triggered (e.g. a handoff to a human).

Rasa Core provides the default implementations of
``action_default_ask_affirmation`` and ``action_default_ask_rephrase``.
The default implementation of ``action_default_ask_rephrase`` action utters
the response template ``utter_ask_rephrase``, so be sure to specify this
template in your domain file.
The implementation of both actions can be overwritten with :ref:`customactions`.

You can specify the core fallback action as well as the ultimate NLU
fallback action as parameters to ``TwoStageFallbackPolicy`` in your
policy configuration file.

.. code-block:: yaml

    policies:
      - name: TwoStageFallbackPolicy
        nlu_threshold: 0.3
        core_threshold: 0.3
        fallback_core_action_name: "action_default_fallback"
        fallback_nlu_action_name: "action_default_fallback"
        deny_suggestion_intent_name: "out_of_scope"


.. include:: feedback.inc
