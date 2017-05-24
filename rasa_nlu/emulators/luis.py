from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

from typing import Any
from typing import Dict
from typing import Text

from rasa_nlu.emulators import NoEmulator


class LUISEmulator(NoEmulator):
    def __init__(self):
        # type: () -> None

        super(LUISEmulator, self).__init__()
        self.name = 'luis'

    def _top_intent(self, data):
        if data.get("intent"):
            return {
                "intent": data["intent"]["name"],
                "score": data["intent"]["confidence"]
            }
        else:
            return None

    def _ranking(self, data):
        if data.get("intent_ranking"):
            return [{"intent": el["name"], "score": el["confidence"]} for el in data["intent_ranking"]]
        else:
            top = self._top_intent(data)
            return [top] if top else []

    def normalise_response_json(self, data):
        # type: (Dict[Text, Any]) -> Dict[Text, Any]
        """Transform data to luis.ai format."""

        top_intent = self._top_intent(data)
        ranking = self._ranking(data)
        return {
            "query": data["text"],
            "topScoringIntent": top_intent,
            "intents": ranking,
            "entities": [
                {
                    "entity": e["value"],
                    "type": e["entity"],
                    "startIndex": None,
                    "endIndex": None,
                    "score": None
                } for e in data["entities"]
                ] if "entities" in data else []
        }
