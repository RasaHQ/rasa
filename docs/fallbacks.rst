.. _fallbacks:

Fallback / Default Actions
==========================



Sometimes you want to fall back to a default action like saying "Sorry, I didn't understand that".
To do this, add the ``FallbackPolicy`` to your policy ensemble.
The default action will be executed if the intent recognition has a confidence below ``nlu_threshold``
or if none of the dialogue policies predict an action with confidence higher than ``core_threshold``

.. code-block:: python

   from rasa_core.policies.fallback import FallbackPolicy
   from rasa_core.policies.keras_policy import KerasPolicy
   from rasa_core.agent import Agent

   fallback = FallbackPolicy(fallback_action_name="utter_default",
                             core_threshold=0.3,
                             nlu_threshold=0.3)

   agent = Agent("domain.yml",
                  policies=[KerasPolicy(), fallback])

