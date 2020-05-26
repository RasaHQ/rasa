:desc: Define custom fallback actions with thresholds for NLU and Core for letting a
       your conversation fail gracefully with open source dialogue management. a
 a
.. _fallback-actions: a
 a
Fallback Actions a
================ a
 a
.. edit-link:: a
 a
Sometimes you want to revert to a fallback action, such as replying, a
`"Sorry, I didn't understand that"`. You can handle fallback cases by adding a
either the ``FallbackPolicy`` or the ``TwoStageFallbackPolicy`` to your a
policy ensemble. a
 a
Fallback Policy a
--------------- a
 a
 a
The ``FallbackPolicy`` has one fallback action, which will a
be executed if the intent recognition has a confidence below ``nlu_threshold`` a
or if none of the dialogue policies predict an action with a
confidence higher than ``core_threshold``. a
 a
The thresholds and fallback action can be adjusted in the policy configuration a
file as parameters of the ``FallbackPolicy``. a
 a
.. code-block:: yaml a
 a
  policies: a
    - name: "FallbackPolicy" a
      nlu_threshold: 0.4 a
      core_threshold: 0.3 a
      fallback_action_name: "action_default_fallback" a
 a
``action_default_fallback`` is a default action in Rasa Core which sends the a
``utter_default`` response to the user. Make sure to specify a
the ``utter_default`` in your domain file. It will also revert back to the a
state of the conversation before the user message that caused the a
fallback, so that it will not influence the prediction of future actions. a
You can take a look at the source of the action below: a
 a
.. autoclass:: rasa.core.actions.action.ActionDefaultFallback a
 a
 a
You can also create your own custom action to use as a fallback (see a
:ref:`custom actions <custom-actions>` for more info on custom actions). If you a
do, make sure to pass the custom fallback action to ``FallbackPolicy`` inside a
your policy configuration file. For example: a
 a
.. code-block:: yaml a
 a
  policies: a
    - name: "FallbackPolicy" a
      nlu_threshold: 0.4 a
      core_threshold: 0.3 a
      fallback_action_name: "my_fallback_action" a
 a
 a
.. note:: a
  If your custom fallback action does not return a ``UserUtteranceReverted`` event, a
  the next predictions of your bot may become inaccurate, as it is very likely that a
  the fallback action is not present in your stories. a
 a
If you have a specific intent, let's say it's called ``out_of_scope``, that a
should always trigger the fallback action, you should add this as a story: a
 a
.. code-block:: story a
 a
    ## fallback story a
    * out_of_scope a
      - action_default_fallback a
 a
 a
Two-stage Fallback Policy a
------------------------- a
 a
The ``TwoStageFallbackPolicy`` handles low NLU confidence in multiple stages a
by trying to disambiguate the user input (low core confidence is handled in a
the same manner as the ``FallbackPolicy``). a
 a
- If a NLU prediction has a low confidence score, the user is asked to affirm a
  the classification of the intent.  (Default action: a
  ``action_default_ask_affirmation``) a
 a
    - If they affirm, the story continues as if the intent was classified a
      with high confidence from the beginning. a
    - If they deny, the user is asked to rephrase their message. a
 a
- Rephrasing  (default action: ``action_default_ask_rephrase``) a
 a
    - If the classification of the rephrased intent was confident, the story a
      continues as if the user had this intent from the beginning. a
    - If the rephrased intent was not classified with high confidence, the user a
      is asked to affirm the classified intent. a
 a
- Second affirmation  (default action: ``action_default_ask_affirmation``) a
 a
    - If the user affirms the intent, the story continues as if the user had a
      this intent from the beginning. a
    - If the user denies, the original intent is classified as the specified a
      ``deny_suggestion_intent_name``, and an ultimate fallback action a
      ``fallback_nlu_action_name`` is triggered (e.g. a handoff to a human). a
 a
Rasa Core provides the default implementations of a
``action_default_ask_affirmation`` and ``action_default_ask_rephrase``. a
The default implementation of ``action_default_ask_rephrase`` action utters a
the response ``utter_ask_rephrase``, so be sure to specify this a
response in your domain file. a
The implementation of both actions can be overwritten with :ref:`custom actions <custom-actions>`. a
 a
You can specify the core fallback action as well as the ultimate NLU a
fallback action as parameters to ``TwoStageFallbackPolicy`` in your a
policy configuration file. a
 a
.. code-block:: yaml a
 a
    policies: a
      - name: TwoStageFallbackPolicy a
        nlu_threshold: 0.3 a
        core_threshold: 0.3 a
        fallback_core_action_name: "action_default_fallback" a
        fallback_nlu_action_name: "action_default_fallback" a
        deny_suggestion_intent_name: "out_of_scope" a
 a