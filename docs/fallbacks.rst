:desc: Define custom fallback actions with thresholds for NLU and Core for letting
       your conversation fail gracefully with open source dialogue management.

.. _fallbacks:

Fallback Actions
==========================



Sometimes you want to fall back to a fallback action like saying
`"Sorry, I didn't understand that"`. To do this, add the
``FallbackPolicy`` to your policy ensemble. The fallback action will
be executed if the intent recognition has a confidence below ``nlu_threshold``
or if none of the dialogue policies predict an action with
confidence higher than ``core_threshold``.

The ``rasa_core.train`` scripts provides parameters to adjust these
thresholds:

+-----------------------+------------------------------------------------------+
| ``--nlu_threshold``   | min confidence needed                                |
|                       | to accept an NLU prediction                          |
+-----------------------+------------------------------------------------------+
| ``--core_threshold``  | min confidence needed                                |
|                       | to accept an action prediction from Rasa Core        |
+-----------------------+------------------------------------------------------+
| ``--fallback_action`` | name of the action to be called if the confidence    |
|                       | of intent / action prediction is below the threshold |
+-----------------------+------------------------------------------------------+

If you want to run this from python, use:

.. code-block:: python

   from rasa_core.policies.fallback import FallbackPolicy
   from rasa_core.policies.keras_policy import KerasPolicy
   from rasa_core.agent import Agent

   fallback = FallbackPolicy(fallback_action_name="action_default_fallback",
                             core_threshold=0.3,
                             nlu_threshold=0.3)

   agent = Agent("domain.yml", policies=[KerasPolicy(), fallback])


``action_default_fallback`` is a default action in Rasa Core, which will send the
``utter_default`` template message to the user. Make sure to specify
this template in your domain file. It will also revert back to the
state of the conversation before the user message that caused the
fallback, so that it will not influence the prediction of future actions.
You can take a look at the source of the action below:

.. autoclass:: rasa_core.actions.action.ActionDefaultFallback


You can also create your own custom action to use as a fallback. If you do, then
make sure to pass the custom fallback action to ``FallbackPolicy`` inside your
policy configuration file. For example:

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

If you have a specific intent that will trigger this, let's say it's
called ``out_of_scope``, then you should add this as a story:

.. code-block:: story

    ## fallback story
    * out_of_scope
      - action_default_fallback


.. include:: feedback.inc
