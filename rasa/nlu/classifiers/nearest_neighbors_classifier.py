import logging
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Text, Dict, List

import joblib
from sentence_transformers import InputExample, losses, SentenceTransformer
from sentence_transformers.datasets import SentenceLabelDataset
from sentence_transformers.losses import BatchHardTripletLossDistanceFunction
from sklearn import preprocessing
from sklearn.metrics import pairwise_distances
from torch.utils.data import DataLoader
import numpy as np

from rasa.engine.storage.resource import Resource
from rasa.engine.storage.storage import ModelStorage
from rasa.engine.recipes.default_recipe import DefaultV1Recipe
from rasa.engine.graph import ExecutionContext, GraphComponent
from rasa.nlu.classifiers import LABEL_RANKING_LENGTH
from rasa.nlu.classifiers.classifier import IntentClassifier
from rasa.shared.nlu.training_data.training_data import TrainingData
from rasa.shared.nlu.training_data.message import Message
from rasa.shared.nlu.constants import TEXT, INTENT
from rasa.utils.tensorflow.constants import RANKING_LENGTH

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Example:
    sentence: str
    label: str
    embedding: np.ndarray


@DefaultV1Recipe.register(
    DefaultV1Recipe.ComponentType.INTENT_CLASSIFIER, is_trainable=True
)
class NearestNeighborsClassifier(IntentClassifier, GraphComponent):
    """Intent classifier using its nearest neighbors."""

    @staticmethod
    def required_packages() -> List[Text]:
        """Any extra python dependencies required for this component to run."""
        return ["sklearn", "sentence_transformers"]

    @staticmethod
    def get_default_config() -> Dict[Text, Any]:
        """The component's default config (see parent class for full docstring)."""
        return {
            "finetune_embeddings": False,
            "finetuning_batch_size": 128,
            "finetuning_epochs": 2,
            "device": "cpu",
            "model": "rasa/LaBSE",
            RANKING_LENGTH: LABEL_RANKING_LENGTH,
        }

    def __init__(
        self,
        config: Dict[Text, Any],
        name: Text,
        model_storage: ModelStorage,
        resource: Resource,
    ) -> None:
        """Construct a new classifier."""
        self.config = {**self.get_default_config(), **config}
        self.name = name
        self.model = None
        self.label_encoder = None
        self.example_embeddings = None

        # We need to use these later when saving the trained component.
        self._model_storage = model_storage
        self._resource = resource

    def finetune_embedding_model(self, sentences: List[str], label_ids: List[int]):
        batch_size = self.config.get("finetuning_batch_size")

        train_examples = [InputExample(texts=[sentence], label=label_id)
                          for sentence, label_id in zip(sentences, label_ids)]
        train_data_sampler = SentenceLabelDataset(train_examples,
                                                  samples_per_label=2)
        batch_size = min(batch_size, len(train_data_sampler))
        train_dataloader = DataLoader(train_data_sampler,
                                      batch_size=batch_size, drop_last=True)

        train_loss = losses.BatchHardTripletLoss(
            model=self.model,
            distance_metric=BatchHardTripletLossDistanceFunction.cosine_distance,
            margin=0.25)
        self.model.fit(train_objectives=[(train_dataloader, train_loss)],
                       epochs=self.config.get("finetuning_epochs", 2),
                       warmup_steps=100)

    def train(self, training_data: TrainingData) -> Resource:
        """Train the intent classifier on a data set."""
        self.label_encoder = preprocessing.LabelEncoder()
        self.model = SentenceTransformer(self.config.get("model"),
                                         device=self.config.get("device"))
        self.example_embeddings = defaultdict(list)

        sentences = [m.get(TEXT) for m in training_data.intent_examples]
        labels = [m.get(INTENT) for m in training_data.intent_examples]
        self.label_encoder.fit(labels)
        label_ids = self.label_encoder.transform(labels)

        if self.config.get("finetune_embeddings", False):
            self.finetune_embedding_model(sentences, label_ids)

        embeddings = self.model.encode(sentences, 16)

        grouped_embeddings = defaultdict(list)
        for idx in range(len(labels)):
            grouped_embeddings[labels[idx]].append(
                Example(sentences[idx], labels[idx], embeddings[idx]))

        for label, examples in grouped_embeddings.items():
            embeddings = [e.embedding for e in examples]
            self.example_embeddings[label] = embeddings

        self.persist()

        return self._resource

    @classmethod
    def create(
        cls,
        config: Dict[Text, Any],
        model_storage: ModelStorage,
        resource: Resource,
        execution_context: ExecutionContext,
    ) -> "NearestNeighborsClassifier":
        """Creates a new untrained component (see parent class for full docstring)."""
        return cls(config, execution_context.node_name, model_storage, resource)

    def process(self, messages: List[Message]) -> List[Message]:
        """Return the most likely intent and its probability for a message."""
        sentences = [m.get(TEXT) for m in messages]
        embeddings = self.model.encode(sentences, 16, show_progress_bar=False)
        for sentence, embedding, message in zip(sentences, embeddings, messages):
            intent_ranking = []
            # go through all labels and their embeddings
            for label, example_embeddings in self.example_embeddings.items():
                distances = pairwise_distances([embedding], example_embeddings,
                                               metric="cosine")
                # use the closest neighbor as the confidence
                intent_ranking.append(
                    {"name": label, "confidence": 1 - np.min(distances)})

            intent_ranking = sorted(intent_ranking, key=lambda p: -p["confidence"])
            intent = intent_ranking[0]
            if self.config[RANKING_LENGTH] > 0:
                intent_ranking = intent_ranking[: self.config[RANKING_LENGTH]]
            message.set(INTENT, intent, add_to_output=True)
            message.set("intent_ranking", intent_ranking, add_to_output=True)
        return messages

    def persist(self) -> None:
        """Persist this model into the passed directory."""
        with self._model_storage.write_to(self._resource) as model_dir:
            path = model_dir / f"{self._resource.name}.joblib"
            model = self.model if self.config["finetune_embeddings"] \
                else self.config["model"]
            joblib.dump([self.label_encoder, model, self.example_embeddings], path)
            logger.debug(f"Saved intent classifier to '{path}'.")

    @classmethod
    def load(
        cls,
        config: Dict[Text, Any],
        model_storage: ModelStorage,
        resource: Resource,
        execution_context: ExecutionContext,
        **kwargs: Any,
    ) -> "NearestNeighborsClassifier":
        """Loads trained component (see parent class for full docstring)."""
        try:
            with model_storage.read_from(resource) as model_dir:
                [encoder, model, example_embeddings] =\
                    joblib.load(model_dir / f"{resource.name}.joblib")
                component = cls(
                    config, execution_context.node_name, model_storage, resource
                )
                component.label_encoder = encoder
                if isinstance(model, str):
                    component.model = \
                        SentenceTransformer(model, device=config.get("device", "cpu"))
                else:
                    component.model = model
                component.example_embeddings = example_embeddings
                return component
        except ValueError:
            logger.debug(
                f"Failed to load {cls.__class__.__name__} from model storage. Resource "
                f"'{resource.name}' doesn't exist."
            )
            return cls.create(config, model_storage, resource, execution_context)

    def process_training_data(self, training_data: TrainingData) -> TrainingData:
        """Process the training data."""
        self.process(training_data.training_examples)
        return training_data
