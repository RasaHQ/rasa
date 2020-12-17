import os
import typing
from typing import Any, Dict, List, Optional, Text, Type

from rasa.nlu.utils.mitie_utils import MitieNLP
from rasa.nlu.tokenizers.tokenizer import Tokenizer
from rasa.nlu.classifiers.classifier import IntentClassifier
from rasa.nlu.components import Component
from rasa.nlu.config import RasaNLUModelConfig
from rasa.nlu.model import Metadata
from rasa.nlu.constants import TOKENS_NAMES
from rasa.shared.nlu.constants import TEXT, INTENT
from rasa.shared.nlu.training_data.training_data import TrainingData
from rasa.shared.nlu.training_data.message import Message
from rasa.shared.exceptions import RasaTrainChunkException
from rasa.utils.tensorflow.data_generator import DataChunkFile

if typing.TYPE_CHECKING:
    import mitie


class MitieIntentClassifier(IntentClassifier):
    """Intent classifier that uses the library MITIE."""

    @classmethod
    def required_components(cls) -> List[Type[Component]]:
        """Specifies which components need to be present in the pipeline."""
        return [MitieNLP, Tokenizer]

    def __init__(
        self,
        component_config: Optional[Dict[Text, Any]] = None,
        classifier: Optional[Any] = None,
    ) -> None:
        """Construct a new intent classifier using the MITIE framework.

        Args:
            component_config: The component configuration.
            classifier: The MITIE classifier.
        """
        super().__init__(component_config)

        self.classifier = classifier

    @classmethod
    def required_packages(cls) -> List[Text]:
        """Specify which python packages need to be installed.

        See parent class for more information.
        """
        return ["mitie"]

    def train_chunk(
        self,
        data_chunk_files: List[DataChunkFile],
        config: Optional[RasaNLUModelConfig] = None,
        **kwargs: Any,
    ) -> None:
        """Trains this component using the list of data chunk files.

        See parent class for more information.
        """
        raise RasaTrainChunkException(
            "This method should neither be called nor implemented in our code."
        )

    def train(
        self,
        training_data: TrainingData,
        config: Optional[RasaNLUModelConfig] = None,
        **kwargs: Any,
    ) -> None:
        """Train this component. See parent class for more information."""
        import mitie

        model_file = kwargs.get("mitie_file")
        if not model_file:
            raise Exception(
                "Can not run MITIE entity extractor without a "
                "language model. Make sure this component is "
                "preceeded by the 'MitieNLP' component."
            )

        trainer = mitie.text_categorizer_trainer(model_file)
        trainer.num_threads = kwargs.get("num_threads", 1)

        for example in training_data.intent_examples:
            tokens = self._tokens_of_message(example)
            trainer.add_labeled_text(tokens, example.get(INTENT))

        if training_data.intent_examples:
            # we can not call train if there are no examples!
            self.classifier = trainer.train()

    def process(self, message: Message, **kwargs: Any) -> None:
        """Process an incoming message."""
        mitie_feature_extractor = kwargs.get("mitie_feature_extractor")
        if not mitie_feature_extractor:
            raise Exception(
                "Failed to train 'MitieFeaturizer'. "
                "Missing a proper MITIE feature extractor."
            )

        if self.classifier:
            token_texts = self._tokens_of_message(message)
            intent, confidence = self.classifier(token_texts, mitie_feature_extractor)
        else:
            # either the model didn't get trained or it wasn't
            # provided with any data
            intent = None
            confidence = 0.0

        message.set(
            "intent", {"name": intent, "confidence": confidence}, add_to_output=True
        )

    @staticmethod
    def _tokens_of_message(message) -> List[Text]:
        return [token.text for token in message.get(TOKENS_NAMES[TEXT], [])]

    @classmethod
    def load(
        cls,
        meta: Dict[Text, Any],
        model_dir: Optional[Text] = None,
        model_metadata: Optional[Metadata] = None,
        cached_component: Optional["MitieIntentClassifier"] = None,
        **kwargs: Any,
    ) -> "MitieIntentClassifier":
        import mitie

        file_name = meta.get("file")

        if not file_name:
            return cls(meta)
        classifier_file = os.path.join(model_dir, file_name)
        if os.path.exists(classifier_file):
            classifier = mitie.text_categorizer(classifier_file)
            return cls(meta, classifier=classifier)
        else:
            return cls(meta)

    def persist(self, file_name: Text, model_dir: Text) -> Dict[Text, Any]:
        """Persist this component to disk for future loading.

        Args:
            file_name: The file name of the model.
            model_dir: The directory to store the model to.

        Returns:
            A dictionary with any information about the stored model.
        """
        if self.classifier:
            file_name = file_name + ".dat"
            classifier_file = os.path.join(model_dir, file_name)
            self.classifier.save_to_disk(classifier_file, pure_model=True)
            return {"file": file_name}
        else:
            return {"file": None}
