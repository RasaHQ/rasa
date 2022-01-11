import logging
from typing import Any, Text, Dict, List, Type

from joblib import dump, load
from scipy.sparse import hstack, vstack, csr_matrix
from sklearn.linear_model import LogisticRegression

from rasa.engine.storage.resource import Resource
from rasa.engine.storage.storage import ModelStorage
from rasa.engine.recipes.default_recipe import DefaultV1Recipe
from rasa.engine.graph import ExecutionContext, GraphComponent
from rasa.nlu.featurizers.featurizer import Featurizer
from rasa.nlu.classifiers.classifier import IntentClassifier
from rasa.nlu.classifiers import LABEL_RANKING_LENGTH
from rasa.shared.nlu.training_data.training_data import TrainingData
from rasa.shared.nlu.training_data.message import Message
from rasa.shared.nlu.constants import TEXT, INTENT


logger = logging.getLogger(__name__)


@DefaultV1Recipe.register(
    DefaultV1Recipe.ComponentType.INTENT_CLASSIFIER, is_trainable=True
)
class LogisticRegressionClassifier(IntentClassifier, GraphComponent):
    @classmethod
    def required_components(cls) -> List[Type]:
        return [Featurizer]

    @staticmethod
    def required_packages() -> List[Text]:
        """Any extra python dependencies required for this component to run."""
        return ["sklearn"]

    @staticmethod
    def get_default_config() -> Dict[Text, Any]:
        return {"class_weight": "balanced", "max_iter": 100, "solver": "lbfgs"}

    def __init__(
        self,
        config: Dict[Text, Any],
        name: Text,
        model_storage: ModelStorage,
        resource: Resource,
    ) -> None:
        self.name = name
        self.clf = LogisticRegression(
            solver=config["solver"],
            max_iter=config["max_iter"],
            class_weight=config["class_weight"],
        )

        # We need to use these later when saving the trained component.
        self._model_storage = model_storage
        self._resource = resource

    def _create_X(self, messages: List[Message]) -> csr_matrix:
        """This method creates a sparse X array that can be used for predicting"""
        X = []
        for e in messages:
            # First element is sequence features, second is sentence features
            sparse_feats = e.get_sparse_features(attribute=TEXT)[1]
            # First element is sequence features, second is sentence features
            dense_feats = e.get_dense_features(attribute=TEXT)[1]
            together = hstack(
                [
                    csr_matrix(sparse_feats.features if sparse_feats else []),
                    csr_matrix(dense_feats.features if dense_feats else []),
                ]
            )
            X.append(together)
        return vstack(X)

    def _create_training_matrix(self, training_data: TrainingData):
        """
        This method creates a scikit-learn compatible (X, y)-pair for training
        the logistic regression model.
        """
        X = []
        y = []
        for e in training_data.training_examples:
            if e.get(INTENT):
                if e.get("text"):
                    # First element is sequence features, second is sentence features
                    sparse_feats = e.get_sparse_features(attribute=TEXT)[1]
                    # First element is sequence features, second is sentence features
                    dense_feats = e.get_dense_features(attribute=TEXT)[1]
                    together = hstack(
                        [
                            csr_matrix(sparse_feats.features if sparse_feats else []),
                            csr_matrix(dense_feats.features if dense_feats else []),
                        ]
                    )
                    X.append(together)
                    y.append(e.get(INTENT))
        return vstack(X), y

    def train(self, training_data: TrainingData) -> Resource:
        X, y = self._create_training_matrix(training_data)
        if X.shape[0] == 0:
            logger.debug(
                f"Cannot train '{self.__class__.__name__}'. No data was provided. "
                f"Skipping training of the classifier."
            )
            return self._resource

        self.clf.fit(X, y)
        self.persist()

        return self._resource

    @classmethod
    def create(
        cls,
        config: Dict[Text, Any],
        model_storage: ModelStorage,
        resource: Resource,
        execution_context: ExecutionContext,
    ) -> GraphComponent:
        return cls(config, execution_context.node_name, model_storage, resource)

    def process(self, messages: List[Message]) -> List[Message]:
        X = self._create_X(messages)
        pred = self.clf.predict(X)
        probas = self.clf.predict_proba(X)
        for idx, message in enumerate(messages):
            intent = {"name": pred[idx], "confidence": probas[idx].max()}
            intents = self.clf.classes_
            intent_info = {
                k: v
                for i, (k, v) in enumerate(zip(intents, probas[idx]))
                if i < LABEL_RANKING_LENGTH
            }
            intent_ranking = [
                {"name": k, "confidence": v} for k, v in intent_info.items()
            ]
            message.set("intent", intent, add_to_output=True)
            message.set("intent_ranking", intent_ranking, add_to_output=True)
        return messages

    def persist(self) -> None:
        with self._model_storage.write_to(self._resource) as model_dir:
            dump(self.clf, model_dir / f"{self.name}.joblib")

    @classmethod
    def load(
        cls,
        config: Dict[Text, Any],
        model_storage: ModelStorage,
        resource: Resource,
        execution_context: ExecutionContext,
    ) -> GraphComponent:
        with model_storage.read_from(resource) as model_dir:
            classifier = load(model_dir / f"{resource.name}.joblib")
            component = cls(
                config, execution_context.node_name, model_storage, resource
            )
            component.clf = classifier
            return component

    def process_training_data(self, training_data: TrainingData) -> TrainingData:
        self.process(training_data.training_examples)
        return training_data

    @classmethod
    def validate_config(cls, config: Dict[Text, Any]) -> None:
        """Validates that the component is configured properly."""
        pass
