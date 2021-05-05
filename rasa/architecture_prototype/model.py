from typing import Dict, Text, Any, Optional, Tuple

import dask

from rasa.architecture_prototype.graph import (
    run_as_dask_graph,
    convert_to_dask_graph,
    graph_component_for_config,
    minimal_dask_graph,
)
from rasa.architecture_prototype.persistence import AbstractModelPersistor
from rasa.architecture_prototype.graph_fingerprinting import TrainingCache
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
        predict_graph_schema: Dict[Text, Any],
    ) -> "Model":
        _fill_defaults(train_graph_schema)
        core_targets = [
            "train_memoization_policy",
            "train_ted_policy",
            "train_rule_policy",
        ]
        nlu_targets = [
            "train_classifier",
            "train_response_selector",
            "train_synonym_mapper",
        ]

        run_as_dask_graph(
            train_graph_schema,
            core_targets + nlu_targets,
            cache=self._cache,
            model_persistor=self._model_persistor,
        )

        _fill_defaults(predict_graph_schema)
        return Model(predict_graph_schema, persistor=self._model_persistor)


class Model:
    def __init__(
        self, rasa_graph: Dict[Text, Any], persistor: "AbstractModelPersistor"
    ) -> None:
        self._persistor = persistor
        self._rasa_graph = rasa_graph
        self._predict_graph = convert_to_dask_graph(
            self._rasa_graph, model_persistor=persistor
        )

    def handle_message(
        self, tracker: DialogueStateTracker, message: Optional[UserMessage]
    ) -> Tuple["PolicyPrediction", Optional[UserUttered]]:
        graph = self._predict_graph.copy()

        # Insert user message into graph
        graph["load_user_message"] = graph_component_for_config(
            "load_user_message",
            self._rasa_graph["load_user_message"],
            {"message": message},
        )

        # Insert dialogue history into graph
        graph["load_history"] = graph_component_for_config(
            "load_history", self._rasa_graph["load_history"], {"tracker": tracker}
        )

        prediction, tracker_with_user_event = dask.get(
            graph, ["select_prediction", "add_parsed_nlu_message"]
        )

        user_event = None
        # if message:
        #     user_event = tracker_with_user_event["add_parsed_nlu_message"].events[-1]

        return prediction["select_prediction"], user_event

    def predict_next_action(
        self, tracker: DialogueStateTracker,
    ) -> Tuple["PolicyPrediction", UserUttered]:
        return self.handle_message(tracker, message=None)

    def get_domain(self) -> Domain:
        domain_graph = minimal_dask_graph(self._predict_graph, targets=["load_domain"])
        return dask.get(domain_graph, "load_domain")["load_domain"]

    def persist(self, target: Text) -> None:
        self._persistor.create_model_package(target, self._predict_graph)

    @classmethod
    def load(self, target: Text, persistor: AbstractModelPersistor) -> "Model":
        graph = persistor.load_model_package(target)

        return Model(graph, persistor)


def _fill_defaults(graph_schema: Dict[Text, Any]):
    for step_name, step_config in graph_schema.items():
        component_class = step_config["uses"]

        if hasattr(component_class, "defaults"):
            step_config["config"] = {
                **component_class.defaults,
                **step_config["config"],
            }
