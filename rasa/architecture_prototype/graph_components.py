import os.path
from pathlib import Path
from typing import Optional, Text, Dict, List

from rasa.architecture_prototype.graph import Persistor
from rasa.core.channels import CollectingOutputChannel, UserMessage
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
