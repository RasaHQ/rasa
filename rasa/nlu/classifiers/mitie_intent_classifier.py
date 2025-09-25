from __future__ import annotations

import logging
import typing
from typing import Any, Dict, List, Optional, Text, Type

# importing mitie at module level to ensure error is raised if mitie is not installed
import mitie

from rasa.engine.graph import ExecutionContext, GraphComponent
from rasa.engine.recipes.default_recipe import DefaultV1Recipe
from rasa.engine.storage.resource import Resource
from rasa.engine.storage.storage import ModelStorage
from rasa.nlu.classifiers.classifier import IntentClassifier
from rasa.nlu.constants import TOKENS_NAMES
from rasa.nlu.featurizers.featurizer import Featurizer
from rasa.nlu.utils.mitie_utils import MitieModel, MitieNLP
from rasa.shared.nlu.constants import INTENT, TEXT
from rasa.shared.nlu.training_data.message import Message
from rasa.shared.nlu.training_data.training_data import TrainingData

if typing.TYPE_CHECKING:
    import mitie

logger = logging.getLogger(__name__)


@DefaultV1Recipe.register(
    DefaultV1Recipe.ComponentType.INTENT_CLASSIFIER,
    is_trainable=True,
    model_from="MitieNLP",
)
class MitieIntentClassifier(GraphComponent, IntentClassifier):
    """Intent classifier which uses the `mitie` library."""

    @classmethod
    def required_components(cls) -> List[Type]:
        """Components that should be included in the pipeline before this component."""
        return [MitieNLP, Featurizer]

    @staticmethod
    def get_default_config() -> Dict[Text, Any]:
        """Returns default config (see parent class for full docstring)."""
        return {"num_threads": 1}

    def __init__(
        self,
        config: Dict[Text, Any],
        model_storage: ModelStorage,
        resource: Resource,
        clf: Optional["mitie.text_categorizer"] = None,
    ) -> None:
        """Constructs a new intent classifier using the MITIE framework."""
        self._config = config
        self._model_storage = model_storage
        self._resource = resource
        self._clf = clf

    @staticmethod
    def required_packages() -> List[Text]:
        """Lists required dependencies (see parent class for full docstring)."""
        return ["mitie"]

    def train(self, training_data: TrainingData, model: MitieModel) -> Resource:
        """Trains classifier.

        Args:
            training_data: The NLU training data.
            model: The loaded mitie model provided by `MitieNLP`.

        Returns:
            The resource locator for the trained classifier.
        """
        import mitie

        trainer = mitie.text_categorizer_trainer(str(model.model_path))
        trainer.num_threads = self._config["num_threads"]

        for example in training_data.intent_examples:
            tokens = self._tokens_of_message(example)
            trainer.add_labeled_text(tokens, example.get(INTENT))

        if training_data.intent_examples:
            # we can not call train if there are no examples!
            clf = trainer.train()
            self._persist(clf)

        return self._resource

    def process(self, messages: List[Message], model: MitieModel) -> List[Message]:
        """Make intent predictions using `mitie`.

        Args:
            messages: The message which the intents should be predicted for.
            model: The loaded mitie model provided by `MitieNLP`.
        """
        for message in messages:
            if self._clf:
                token_strs = self._tokens_of_message(message)
                intent, confidence = self._clf(token_strs, model.word_feature_extractor)
            else:
                # either the model didn't get trained or it wasn't
                # provided with any data
                intent = None
                confidence = 0.0

            message.set(
                "intent", {"name": intent, "confidence": confidence}, add_to_output=True
            )

        return messages

    @staticmethod
    def _tokens_of_message(message: Message) -> List[Text]:
        return [token.text for token in message.get(TOKENS_NAMES[TEXT], [])]

    @classmethod
    def create(
        cls,
        config: Dict[Text, Any],
        model_storage: ModelStorage,
        resource: Resource,
        execution_context: ExecutionContext,
    ) -> MitieIntentClassifier:
        """Creates component for training see parent class for full docstring)."""
        return cls(config, model_storage, resource)

    @classmethod
    def load(
        cls,
        config: Dict[Text, Any],
        model_storage: ModelStorage,
        resource: Resource,
        execution_context: ExecutionContext,
        **kwargs: Any,
    ) -> MitieIntentClassifier:
        """Loads component for inference see parent class for full docstring)."""
        import mitie

        text_categorizer = None

        try:
            with model_storage.read_from(resource) as directory:
                text_categorizer = mitie.text_categorizer(str(directory / "model.dat"))
        except (
            ValueError,
            Exception,
        ):  # the latter is thrown by the `mitie.text_categorizer`
            logger.warning(
                f"Failed to load {cls.__class__.__name__} from model storage. Resource "
                f"'{resource.name}' doesn't exist."
            )

        return cls(config, model_storage, resource, text_categorizer)

    def _persist(self, text_categorizer: "mitie.text_categorizer") -> None:
        """Persists trained model (see parent class for full docstring)."""
        with self._model_storage.write_to(self._resource) as directory:
            classifier_file = directory / "model.dat"
            text_categorizer.save_to_disk(str(classifier_file), pure_model=True)
