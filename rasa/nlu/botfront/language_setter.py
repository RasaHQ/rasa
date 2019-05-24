from typing import Any
from typing import Optional
from typing import Text
from typing import Dict
from rasa.nlu.components import Component
from rasa.nlu.training_data import Message
from rasa.nlu.model import Metadata


class LanguageSetter(Component):
    name = 'LanguageSetter'

    def __init__(self,
                 component_config: Optional[Dict[Text, Any]] = None,
                 ) -> None:
        super(LanguageSetter, self).__init__(component_config)
        self.component_config = component_config

    def process(self, message, **kwargs):
        # type: (Message, **Any) -> None

        message.set("language", self.component_config["language"], add_to_output=True)

    @classmethod
    def load(cls,
             component_meta: Dict[Text, Any],
             model_dir: Text = None,
             model_metadata: Metadata = None,
             cached_component: Optional["LanguageSetter"] = None,
             **kwargs: Any
             ) -> "LanguageSetter":
        return cls(component_meta)
