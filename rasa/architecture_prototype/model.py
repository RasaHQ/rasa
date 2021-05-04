from typing import Dict, List, Text, Any, Optional, Tuple

import dask
import dask.threaded

from rasa.architecture_prototype.graph import (
    run_as_dask_graph,
    convert_to_dask_graph,
    graph_component_for_config,
    minimal_dask_graph,
    run_dask_graph,
)
from rasa.architecture_prototype.persistence import AbstractModelPersistor
from rasa.architecture_prototype.graph_fingerprinting import (
    TrainingCache,
    dask_graph_to_fingerprint_graph,
    prune_graph_schema,
)
from rasa.core.channels import UserMessage
from rasa.core.policies.policy import PolicyPrediction
from rasa.shared.core.domain import Domain
from rasa.shared.core.events import UserUttered
from rasa.shared.core.trackers import DialogueStateTracker


class ModelTrainer:
    def __init__(
        self, model_persistor: AbstractModelPersistor, cache: TrainingCache,
    ) -> None:
        self._model_persistor = model_persistor
        self._cache = cache

    def train(
        self,
        # TODO: add "project"
        train_graph_schema: Dict[Text, Any],
        train_graph_targets: List[Text],
        predict_graph_schema: Dict[Text, Any],
    ) -> "Model":
        _fill_defaults(train_graph_schema)
        dask_graph = convert_to_dask_graph(
            train_graph_schema, cache=self._cache, model_persistor=self._model_persistor
        )
        fingerprint_graph = dask_graph_to_fingerprint_graph(dask_graph, self._cache)
        fingerprint_statuses = run_dask_graph(fingerprint_graph, train_graph_targets)
        pruned_graph_schema = prune_graph_schema(
            train_graph_schema, train_graph_targets, fingerprint_statuses, self._cache,
        )

        run_as_dask_graph(
            pruned_graph_schema,
            train_graph_targets,
            cache=self._cache,
            model_persistor=self._model_persistor,
        )

        _fill_defaults(predict_graph_schema)
        # TODO: this loads the models again
        return Model(
            predict_graph_schema,
            train_graph_schema,
            train_graph_targets,
            self._cache,
            persistor=self._model_persistor
        )


class Model:
    def __init__(
        self,
        predict_graph_schema: Dict[Text, Any],
        train_graph_schema: Dict[Text, Any],
        train_graph_targets: List[Text],
        cache: "TrainingCache",
        persistor: "AbstractModelPersistor",
    ) -> None:
        self._persistor = persistor
        self._predict_graph_schema = predict_graph_schema
        self._dask_graph = convert_to_dask_graph(
            self._predict_graph_schema, model_persistor=persistor
        )
        self._train_graph_schema = train_graph_schema
        self._train_graph_targets = train_graph_targets
        self._cache = cache

    def handle_message(
        self, tracker: DialogueStateTracker, message: Optional[UserMessage]
    ) -> Tuple["PolicyPrediction", Optional[UserUttered]]:
        graph = self._dask_graph.copy()

        # Insert user message into graph
        graph["load_user_message"] = graph_component_for_config(
            "load_user_message",
            self._predict_graph_schema["load_user_message"],
            {"message": message},
        )

        # Insert dialogue history into graph
        graph["load_history"] = graph_component_for_config(
            "load_history", self._predict_graph_schema["load_history"], {"tracker": tracker}
        )

        results = run_dask_graph(graph, ["select_prediction", "add_parsed_nlu_message"])

        user_event = None
        if message:
            user_event = results["add_parsed_nlu_message"].events[-1]

        return results["select_prediction"], user_event

    def predict_next_action(
        self, tracker: DialogueStateTracker,
    ) -> Tuple["PolicyPrediction", UserUttered]:
        return self.handle_message(tracker, message=None)

    def get_domain(self) -> Domain:
        domain_graph = minimal_dask_graph(self._dask_graph, targets=["load_domain"])
        return dask.threaded.get(domain_graph, "load_domain")["load_domain"]

    def persist(self, target: Text) -> None:
        self._persistor.create_model_package(
            target,
            self._predict_graph_schema,
            self._train_graph_schema,
            self._train_graph_targets,
            self._cache,
        )

    @classmethod
    def load(self, target: Text, persistor: AbstractModelPersistor) -> "Model":
        return persistor.load_model_package(target)

    def train(self) -> "Model":
        trainer = ModelTrainer(self._persistor, self._cache)
        return trainer.train(self._train_graph_schema, self._train_graph_targets, self._predict_graph_schema)


def _fill_defaults(graph_schema: Dict[Text, Any]):
    for step_name, step_config in graph_schema.items():
        component_class = step_config["uses"]

        if hasattr(component_class, "defaults"):
            step_config["config"] = {
                **component_class.defaults,
                **step_config["config"],
            }
