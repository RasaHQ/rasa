import os.path
from collections import ChainMap
import inspect
from pathlib import Path
from typing import Any, Text, Dict, List, Union, Optional

import dask

from rasa.core.channels import UserMessage, CollectingOutputChannel
from rasa.nlu.persistor import Persistor
from rasa.shared.constants import DEFAULT_DATA_PATH, DEFAULT_DOMAIN_PATH
from rasa.shared.core.constants import ACTION_LISTEN_NAME
from rasa.shared.core.domain import Domain
from rasa.shared.core.events import ActionExecuted, UserUttered
from rasa.shared.core.generator import TrackerWithCachedStates
from rasa.shared.core.trackers import DialogueStateTracker
from rasa.shared.core.training_data.structures import StoryGraph
from rasa.shared.importers.importer import TrainingDataImporter
from rasa.shared.nlu.constants import ACTION_NAME, ACTION_TEXT, INTENT, TEXT
from rasa.shared.nlu.training_data.message import Message
from rasa.shared.nlu.training_data.training_data import TrainingData
import rasa.shared.utils.common
import rasa.utils.common
import rasa.core.training


class ProjectReader:
    def __init__(self, project: Text) -> None:
        self._project = project

    def load_importer(self) -> TrainingDataImporter:
        return TrainingDataImporter.load_from_dict(
            domain_path=str(Path(self._project, DEFAULT_DOMAIN_PATH)),
            training_data_paths=[os.path.join(self._project, DEFAULT_DATA_PATH)],
        )


class TrainingDataReader(ProjectReader):
    def read(self) -> TrainingData:
        importer = self.load_importer()
        return rasa.utils.common.run_in_loop(importer.get_nlu_data())


class DomainReader(ProjectReader):
    def __init__(
        self,
        project: Optional[Text] = None,
        persistor: Optional["Persistor"] = None,
        resource_name: Optional[Text] = None,
    ) -> None:
        super().__init__(project)
        self._persistor = persistor
        self._resource_name = resource_name

    def read(self) -> Domain:
        importer = self.load_importer()
        domain = rasa.utils.common.run_in_loop(importer.get_domain())

        target_file = self._persistor.file_for("domain.yml")
        domain.persist(target_file)

        return domain

    def provide(self) -> Domain:
        filename = self._persistor.get_resource(self._resource_name, "domain.yml")
        return Domain.load(filename)


class StoryGraphReader(ProjectReader):
    def read(self) -> StoryGraph:
        importer = self.load_importer()

        return rasa.utils.common.run_in_loop(importer.get_stories())


class TrackerGenerator:
    def generate(
        self, domain: Domain, story_graph: StoryGraph
    ) -> List[TrackerWithCachedStates]:
        generated_coroutine = rasa.core.training.load_data(story_graph, domain,)
        return rasa.utils.common.run_in_loop(generated_coroutine)


class StoryToTrainingDataConverter:
    def convert(self, story_graph: StoryGraph) -> TrainingData:
        messages = []
        for tracker in story_graph.story_steps:
            for event in tracker.events:
                if isinstance(event, (ActionExecuted, UserUttered)):
                    messages.append(Message(data=event.as_sub_state()))

        return TrainingData(training_examples=messages)


class MessageToE2EFeatureConverter:
    def convert(self, training_data: TrainingData) -> Dict[Text, Message]:
        additional_features = {}
        for message in training_data.training_examples:
            key = next(
                k
                for k in message.data.keys()
                if k in {ACTION_NAME, ACTION_TEXT, INTENT, TEXT}
            )
            additional_features[key] = message

        return additional_features


class MessageCreator:
    def __init__(self, text):
        self._text = text

    def create(self) -> UserMessage:
        return UserMessage(text=self._text, output_channel=CollectingOutputChannel())


class TrackerLoader:
    def __init__(self, tracker: DialogueStateTracker) -> None:
        self._tracker = tracker

    def load(self) -> DialogueStateTracker:
        return self._tracker


class NLUMessageConverter:
    def convert(self, message: UserMessage) -> Message:
        return Message.build(message.text)


class NLUPredictionToHistoryAdder:
    def merge(
        self,
        tracker: DialogueStateTracker,
        initial_user_message: UserMessage,
        parsed_message: Message,
        domain: Domain,
    ) -> DialogueStateTracker:
        parse_data = parsed_message.as_dict()
        if tracker.latest_action_name == ACTION_LISTEN_NAME:
            tracker.update(
                UserUttered(
                    initial_user_message.text,
                    parse_data["intent"],
                    parse_data["entities"],
                    parse_data,
                    input_channel=initial_user_message.input_channel,
                    message_id=initial_user_message.message_id,
                    metadata=initial_user_message.metadata,
                ),
                domain,
            )
        return tracker


class RasaComponent:
    def __init__(
        self,
        component_class: Any,
        config: Dict[Text, Any],
        fn_name: Text,
        node_name: Text,
        inputs: Dict[Text, Text],
        constructor_name: Text = None,
        eager: bool = True,
        persist: bool = True,
    ) -> None:
        self._eager = eager
        self._inputs = inputs
        self._constructor_name = constructor_name
        self._component_class = component_class
        self._fn_name = fn_name
        self._run_fn = getattr(self._component_class, fn_name)
        self._component = None
        self._node_name = node_name
        self._persist = persist
        self._config = config

        if self._constructor_name:
            self._constructor_fn = getattr(
                self._component_class, self._constructor_name
            )
        else:
            self._constructor_fn = self._component_class

        input_names = list(inputs.keys())
        self.validate_params_in_inputs(input_names, self._run_fn)
        if not eager:
            self.validate_params_in_inputs(input_names, self._constructor_fn)

        if self._eager:
            self.create_component(**self._config)

    def validate_params_in_inputs(self, input_names, func):
        params = inspect.signature(func).parameters
        for param_name, param in params.items():
            if param_name in ["self", "args", "kwargs", "persistor"]:
                continue
            if param.default is inspect._empty:
                if param_name not in input_names:
                    raise ValueError(
                        f"{param_name} for function {func} is missing from inputs"
                    )

    def __call__(self, *args: Any) -> Dict[Text, Any]:
        received_inputs = dict(ChainMap(*args))
        kwargs = {}
        for input, input_node in self._inputs.items():
            kwargs[input] = received_inputs[input_node]

        if not self._eager:
            const_kwargs = rasa.shared.utils.common.minimal_kwargs(
                kwargs, self._constructor_fn
            )
            self.create_component(**const_kwargs)

        run_kwargs = kwargs

        if "kwargs" not in rasa.shared.utils.common.arguments_of(self._run_fn):
            run_kwargs = rasa.shared.utils.common.minimal_kwargs(kwargs, self._run_fn)

        return {self._node_name: self._run_fn(self._component, **run_kwargs)}

    def create_component(self, **const_kwargs: Any) -> None:
        if self._persist:
            const_kwargs["persistor"] = Persistor(
                self._node_name, parent_dir=Path("model")
            )
        self._component = self._constructor_fn(**const_kwargs)

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, RasaComponent):
            return NotImplemented

        return (
            self._node_name == other._node_name
            and self._component_class == other._component_class
            and self._config == other._config
            and self._fn_name == other._fn_name
        )

    def __repr__(self) -> Text:
        return f"{self._component_class}.{self._fn_name}"


class Persistor:
    def __init__(self, node_name: Text, parent_dir: Path) -> None:
        self._node_name = node_name
        self._parent_dir = parent_dir
        self._dir_for_node = Path(parent_dir / node_name)

    def file_for(self, filename: Text) -> Text:
        self._dir_for_node.mkdir(exist_ok=True)
        return str(self._dir_for_node / filename,)

    def directory_for(self, dir_name: Text) -> Text:
        self._dir_for_node.mkdir(exist_ok=True)
        directory = self._dir_for_node / dir_name
        directory.mkdir()
        return str(directory)

    def get_resource(self, resource_name, filename) -> Text:
        return str(Path(self._parent_dir, resource_name, filename))

    def resource_name(self) -> Text:
        return self._node_name


def run_as_dask_graph(
    rasa_graph: Dict[Text, Any], target_names: Union[Text, List[Text]]
) -> Dict[Text, Any]:
    dask_graph = convert_to_dask_graph(rasa_graph)
    return dict(ChainMap(*dask.get(dask_graph, target_names)))


def convert_to_dask_graph(rasa_graph: Dict[Text, Any]):
    dsk = {}
    for step_name, step_config in rasa_graph.items():
        dsk[step_name] = (
            RasaComponent(
                node_name=step_name,
                component_class=step_config["uses"],
                config=step_config["config"],
                fn_name=step_config["fn"],
                inputs=step_config["needs"],
                constructor_name=step_config.get("constructor_name"),
                eager=step_config.get("eager", True),
                persist=step_config.get("persist", True),
            ),
            *step_config["needs"].values(),
        )
    return dsk


def fill_defaults(graph: Dict[Text, Any]):
    for step_name, step_config in graph.items():
        component_class = step_config["uses"]

        if hasattr(component_class, "defaults"):
            defaults = component_class.defaults
            step_config["config"].update(defaults)
