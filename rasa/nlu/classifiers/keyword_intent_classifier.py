from __future__ import annotations
import logging
import re
from typing import Any, Dict, Optional, Text, List

from rasa.engine.graph import GraphComponent, ExecutionContext
from rasa.engine.recipes.default_recipe import DefaultV1Recipe
from rasa.engine.storage.resource import Resource
from rasa.engine.storage.storage import ModelStorage
from rasa.shared.constants import DOCS_URL_COMPONENTS
from rasa.nlu.classifiers.classifier import IntentClassifier
from rasa.shared.nlu.constants import (
    INTENT,
    TEXT,
    INTENT_NAME_KEY,
    PREDICTED_CONFIDENCE_KEY,
)
import rasa.shared.utils.io
from rasa.shared.nlu.training_data.training_data import TrainingData
from rasa.shared.nlu.training_data.message import Message

logger = logging.getLogger(__name__)


@DefaultV1Recipe.register(
    DefaultV1Recipe.ComponentType.INTENT_CLASSIFIER, is_trainable=True
)
class KeywordIntentClassifier(GraphComponent, IntentClassifier):
    """Intent classifier using simple keyword matching.

    The classifier takes a list of keywords and associated intents as an input.
    An input sentence is checked for the keywords and the intent is returned.
    """

    @staticmethod
    def get_default_config() -> Dict[Text, Any]:
        """The component's default config (see parent class for full docstring)."""
        return {"case_sensitive": True}

    def __init__(
        self,
        config: Dict[Text, Any],
        model_storage: ModelStorage,
        resource: Resource,
        execution_context: ExecutionContext,
        intent_keyword_map: Optional[Dict] = None,
    ) -> None:
        """Creates classifier."""
        self.component_config = config
        self._model_storage = model_storage
        self._resource = resource
        self._execution_context = execution_context

        self.case_sensitive = self.component_config.get("case_sensitive")
        self.intent_keyword_map = intent_keyword_map or {}

    @classmethod
    def create(
        cls,
        config: Dict[Text, Any],
        model_storage: ModelStorage,
        resource: Resource,
        execution_context: ExecutionContext,
    ) -> KeywordIntentClassifier:
        """Creates a new untrained component (see parent class for full docstring)."""
        return cls(config, model_storage, resource, execution_context)

    def train(self, training_data: TrainingData) -> Resource:
        """Trains the intent classifier on a data set."""
        duplicate_examples = set()
        for ex in training_data.intent_examples:
            if (
                ex.get(TEXT) in self.intent_keyword_map.keys()
                and ex.get(INTENT) != self.intent_keyword_map[ex.get(TEXT)]
            ):
                duplicate_examples.add(ex.get(TEXT))
                rasa.shared.utils.io.raise_warning(
                    f"Keyword '{ex.get(TEXT)}' is a keyword to trigger intent "
                    f"'{self.intent_keyword_map[ex.get(TEXT)]}' and also "
                    f"intent '{ex.get(INTENT)}', it will be removed "
                    f"from the list of keywords for both of them. "
                    f"Remove (one of) the duplicates from the training data.",
                    docs=DOCS_URL_COMPONENTS + "#keyword-intent-classifier",
                )
            else:
                self.intent_keyword_map[ex.get(TEXT)] = ex.get(INTENT)
        for keyword in duplicate_examples:
            self.intent_keyword_map.pop(keyword)
            logger.debug(
                f"Removed '{keyword}' from the list of keywords because it was "
                "a keyword for more than one intent."
            )

        self._validate_keyword_map()
        self.persist()
        return self._resource

    def _validate_keyword_map(self) -> None:
        re_flag = 0 if self.case_sensitive else re.IGNORECASE

        ambiguous_mappings = []
        for keyword1, intent1 in self.intent_keyword_map.items():
            for keyword2, intent2 in self.intent_keyword_map.items():
                if (
                    re.search(r"\b" + keyword1 + r"\b", keyword2, flags=re_flag)
                    and intent1 != intent2
                ):
                    ambiguous_mappings.append((intent1, keyword1))
                    rasa.shared.utils.io.raise_warning(
                        f"Keyword '{keyword1}' is a keyword of intent '{intent1}', "
                        f"but also a substring of '{keyword2}', which is a "
                        f"keyword of intent '{intent2}."
                        f" '{keyword1}' will be removed from the list of keywords.\n"
                        f"Remove (one of) the conflicting keywords from the"
                        f" training data.",
                        docs=DOCS_URL_COMPONENTS + "#keyword-intent-classifier",
                    )
        for intent, keyword in ambiguous_mappings:
            self.intent_keyword_map.pop(keyword)
            logger.debug(
                f"Removed keyword '{keyword}' from intent "
                f"'{intent}' because it matched a "
                f"keyword of another intent."
            )

    def process(self, messages: List[Message]) -> List[Message]:
        """Sets the message intent and add it to the output if it exists."""
        for message in messages:
            intent_name = self._map_keyword_to_intent(message.get(TEXT))

            confidence = 0.0 if intent_name is None else 1.0
            intent = {
                INTENT_NAME_KEY: intent_name,
                PREDICTED_CONFIDENCE_KEY: confidence,
            }

            if message.get(INTENT) is None or intent_name is not None:
                message.set(INTENT, intent, add_to_output=True)

        return messages

    def _map_keyword_to_intent(self, text: Text) -> Optional[Text]:
        re_flag = 0 if self.case_sensitive else re.IGNORECASE

        for keyword, intent in self.intent_keyword_map.items():
            if re.search(r"\b" + keyword + r"\b", text, flags=re_flag):
                logger.debug(
                    f"KeywordClassifier matched keyword '{keyword}' to"
                    f" intent '{intent}'."
                )
                return intent

        logger.debug("KeywordClassifier did not find any keywords in the message.")
        return None

    def persist(self) -> None:
        """Persist this model into the passed directory."""
        with self._model_storage.write_to(self._resource) as model_dir:
            file_name = f"{self.__class__.__name__}.json"
            keyword_file = model_dir / file_name
            rasa.shared.utils.io.dump_obj_as_json_to_file(
                keyword_file, self.intent_keyword_map
            )

    @classmethod
    def load(
        cls,
        config: Dict[Text, Any],
        model_storage: ModelStorage,
        resource: Resource,
        execution_context: ExecutionContext,
        **kwargs: Any,
    ) -> KeywordIntentClassifier:
        """Loads trained component (see parent class for full docstring)."""
        try:
            with model_storage.read_from(resource) as model_dir:
                keyword_file = model_dir / f"{cls.__name__}.json"
                intent_keyword_map = rasa.shared.utils.io.read_json_file(keyword_file)
        except ValueError:
            logger.warning(
                f"Failed to load {cls.__class__.__name__} from model storage. Resource "
                f"'{resource.name}' doesn't exist."
            )
            intent_keyword_map = None

        return cls(
            config, model_storage, resource, execution_context, intent_keyword_map
        )
