from typing import Any, List, Type, Text, Dict, Union

from rasa.constants import DEFAULT_NLU_FALLBACK_INTENT_NAME
from rasa.core.constants import DEFAULT_NLU_FALLBACK_THRESHOLD
from rasa.nlu.classifiers.classifier import IntentClassifier
from rasa.nlu.components import Component
from rasa.nlu.training_data import Message
from rasa.nlu.constants import INTENT_RANKING_KEY, INTENT, INTENT_CONFIDENCE_KEY

THRESHOLD_KEY = "threshold"
FALLBACK_INTENT_NAME_KEY = "fallback_intent_name"


class FallbackClassifier(Component):

    # please make sure to update the docs when changing a default parameter
    defaults = {
        # ## Architecture of the used neural network
        # If all intent confidence scores are beyond this threshold, set the current
        # intent to `FALLBACK_INTENT_NAME`
        THRESHOLD_KEY: DEFAULT_NLU_FALLBACK_THRESHOLD,
        # The intent which is used to signal that the NLU confidence was below the
        # threshold.
        FALLBACK_INTENT_NAME_KEY: DEFAULT_NLU_FALLBACK_INTENT_NAME,
    }

    @classmethod
    def required_components(cls) -> List[Type[Component]]:
        return [IntentClassifier]

    def process(self, message: Message, **kwargs: Any) -> None:
        """Process an incoming message.

        This is the components chance to process an incoming
        message. The component can rely on
        any context attribute to be present, that gets created
        by a call to :meth:`rasa.nlu.components.Component.create`
        of ANY component and
        on any context attributes created by a call to
        :meth:`rasa.nlu.components.Component.process`
        of components previous to this one.

        Args:
            message: The :class:`rasa.nlu.training_data.message.Message` to process.

        """

        if not self._should_fallback(message):
            return

        message.data[INTENT] = self._fallback_intent()
        message.data[INTENT_RANKING_KEY].insert(0, self._fallback_intent())

    def _should_fallback(self, message: Message) -> bool:
        return (
            message.data[INTENT].get(INTENT_CONFIDENCE_KEY)
            < self.component_config[THRESHOLD_KEY]
        )

    def _fallback_intent(self) -> Dict[Text, Union[Text, float]]:
        return {
            "name": self.component_config[FALLBACK_INTENT_NAME_KEY],
            # TODO: Re-consider how we represent the confidence here
            INTENT_CONFIDENCE_KEY: 1.0,
        }
