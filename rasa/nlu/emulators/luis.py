from typing import Any, Dict, Text

from rasa_nlu.emulators import NoEmulator


class LUISEmulator(NoEmulator):
    def __init__(self) -> None:

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
            return [{"intent": el["name"], "score": el["confidence"]}
                    for el in data["intent_ranking"]]
        else:
            top = self._top_intent(data)
            return [top] if top else []

    def normalise_response_json(self, data: Dict[Text, Any]) -> Dict[Text, Any]:
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
                    "startIndex": e.get("start"),
                    "endIndex": (e["end"] - 1) if "end" in e else None,
                    "score": e.get("confidence")
                } for e in data["entities"]
            ] if "entities" in data else []
        }
