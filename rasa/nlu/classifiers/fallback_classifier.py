from typing import Any, List, Type, Text, Dict, Union

from rasa.constants import DEFAULT_NLU_FALLBACK_INTENT_NAME
from rasa.core.constants import DEFAULT_NLU_FALLBACK_THRESHOLD
from rasa.nlu.classifiers.classifier import IntentClassifier
from rasa.nlu.components import Component
from rasa.nlu.training_data import Message
from rasa.nlu.constants import INTENT_RANKING_KEY, INTENT, INTENT_CONFIDENCE_KEY

THRESHOLD_KEY = "threshold"


class FallbackClassifier(Component):

    # please make sure to update the docs when changing a default parameter
    defaults = {
        # If all intent confidence scores are beyond this threshold, set the current
        # intent to `FALLBACK_INTENT_NAME`
        THRESHOLD_KEY: DEFAULT_NLU_FALLBACK_THRESHOLD
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

        message.data[INTENT] = _fallback_intent()
        message.data[INTENT_RANKING_KEY].insert(0, _fallback_intent())

    def _should_fallback(self, message: Message) -> bool:
        return (
            message.data[INTENT].get(INTENT_CONFIDENCE_KEY)
            < self.component_config[THRESHOLD_KEY]
        )


def _fallback_intent() -> Dict[Text, Union[Text, float]]:
    return {
        "name": DEFAULT_NLU_FALLBACK_INTENT_NAME,
        # TODO: Re-consider how we represent the confidence here
        INTENT_CONFIDENCE_KEY: 1.0,
    }
