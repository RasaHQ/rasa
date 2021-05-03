import asyncio
import os.path
from pathlib import Path
from typing import Optional, Text, Dict, List, Union

from rasa.architecture_prototype.graph import Persistor
from rasa.core.channels import CollectingOutputChannel, UserMessage
from rasa.shared.constants import DEFAULT_DATA_PATH, DEFAULT_DOMAIN_PATH
from rasa.shared.core.domain import Domain
from rasa.shared.core.events import ActionExecuted, UserUttered, Event
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


class GraphComponentMetaclass(type):
    """Metaclass with `name` class property."""

    @property
    def name(cls) -> Text:
        """The name property is a function of the class - its __name__."""
        return cls.__name__


class GraphComponent(metaclass=GraphComponentMetaclass):
    pass


class ProjectReader(GraphComponent):
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
        persistor: Optional[Persistor] = None,
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


class TrackerGenerator(GraphComponent):
    def generate(
        self, domain: Domain, story_graph: StoryGraph
    ) -> List[TrackerWithCachedStates]:
        generated_coroutine = rasa.core.training.load_data(story_graph, domain,)
        return rasa.utils.common.run_in_loop(generated_coroutine)


class StoryToTrainingDataConverter(GraphComponent):
    def convert_for_training(self, story_graph: StoryGraph) -> TrainingData:
        messages = []
        for step in story_graph.story_steps:
            messages += self._convert_tracker_to_messages(step.events)

        # Workaround: add at least one end to end message to initialize
        # the `CountVectorizer` for e2e. Alternatives: Store information or simply config
        messages.append(
            Message(
                data=UserUttered(
                    text="hi", use_text_for_featurization=True
                ).as_sub_state()
            )
        )
        return TrainingData(training_examples=messages)

    def _convert_tracker_to_messages(self, events: List[Event]) -> List[Message]:
        messages = []
        for event in events:
            if isinstance(event, ActionExecuted):
                messages.append(Message(data=event.as_sub_state()))

            if isinstance(event, UserUttered):
                if event.use_text_for_featurization is None:
                    event.use_text_for_featurization = False
                    messages.append(Message(data=event.as_sub_state()))

                    event.use_text_for_featurization = True
                    messages.append(Message(data=event.as_sub_state()))

                    event.use_text_for_featurization = None
                else:
                    messages.append(Message(data=event.as_sub_state()))

        return messages

    def convert_for_inference(self, tracker: DialogueStateTracker) -> List[Message]:
        return self._convert_tracker_to_messages(tracker.events)


class MessageToE2EFeatureConverter(GraphComponent):
    def convert(self, messages: Union[TrainingData, List[Message]]) -> Dict[Text, Message]:
        if isinstance(messages, TrainingData):
            messages = messages.training_examples
        additional_features = {}
        for message in messages:
            key = next(
                k
                for k in message.data.keys()
                if k in {ACTION_NAME, ACTION_TEXT, INTENT, TEXT}
            )
            additional_features[key] = message

        return additional_features


class MessageCreator(GraphComponent):
    def __init__(self, message: Optional[UserMessage]) -> None:
        self._message = message

    def create(self) -> Optional[UserMessage]:
        return self._message


class TrackerLoader(GraphComponent):
    def __init__(self, tracker: DialogueStateTracker) -> None:
        self._tracker = tracker

    def load(self) -> DialogueStateTracker:
        return self._tracker


class NLUMessageConverter(GraphComponent):
    def convert(self, message: Optional[UserMessage]) -> List[Message]:
        if message:
            return [Message.build(message.text)]
        return []


class NLUPredictionToHistoryAdder(GraphComponent):
    def merge(
        self,
        tracker: DialogueStateTracker,
        domain: Domain,
        initial_user_message: Optional[UserMessage] = None,
        parsed_messages: List[Message] = None,
    ) -> DialogueStateTracker:
        for parsed_message in parsed_messages:
            parse_data = parsed_message.as_dict(only_output_properties=True)

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


components = [
    ProjectReader,
    TrainingDataReader,
    DomainReader,
    StoryGraphReader,
    TrackerGenerator,
    StoryToTrainingDataConverter,
    MessageToE2EFeatureConverter,
    MessageCreator,
    TrackerLoader,
    NLUMessageConverter,
    NLUPredictionToHistoryAdder,
]
registry = {c.name: c for c in components}


class GraphComponentNotFound(Exception):
    pass


def load_graph_component(name: Text):
    if name not in registry:
        raise GraphComponentNotFound()

    return registry[name]
