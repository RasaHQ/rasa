import logging
from abc import ABC, abstractmethod
from functools import reduce
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Set,
    Text,
    Tuple,
    Type,
    Union,
    cast,
)

import importlib_resources

import rasa.shared.constants
import rasa.shared.core.constants
import rasa.shared.data
import rasa.shared.utils.common
import rasa.shared.utils.io
from rasa.shared.constants import CONFIG_ADDITIONAL_LANGUAGES_KEY, CONFIG_LANGUAGE_KEY
from rasa.shared.core.domain import (
    IS_RETRIEVAL_INTENT_KEY,
    KEY_ACTIONS,
    KEY_E2E_ACTIONS,
    KEY_INTENTS,
    KEY_RESPONSES,
    KEY_SLOTS,
    Domain,
)
from rasa.shared.core.events import ActionExecuted, UserUttered
from rasa.shared.core.flows import FlowsList
from rasa.shared.core.slots import StrictCategoricalSlot
from rasa.shared.core.training_data.structures import StoryGraph
from rasa.shared.exceptions import RasaException
from rasa.shared.nlu.constants import ACTION_NAME, ENTITIES
from rasa.shared.nlu.training_data.message import Message
from rasa.shared.nlu.training_data.training_data import TrainingData
from rasa.shared.utils.common import cached_method
from rasa.shared.utils.yaml import read_config_file

logger = logging.getLogger(__name__)


class TrainingDataImporter(ABC):
    """Common interface for different mechanisms to load training data."""

    @abstractmethod
    def __init__(
        self,
        config_file: Optional[Text] = None,
        domain_path: Optional[Text] = None,
        training_data_paths: Optional[Union[List[Text], Text]] = None,
        **kwargs: Any,
    ) -> None:
        """Initialise the importer."""
        ...

    @abstractmethod
    def get_domain(self) -> Domain:
        """Retrieves the domain of the bot.

        Returns:
            Loaded `Domain`.
        """
        ...

    @abstractmethod
    def get_stories(self, exclusion_percentage: Optional[int] = None) -> StoryGraph:
        """Retrieves the stories that should be used for training.

        Args:
            exclusion_percentage: Amount of training data that should be excluded.

        Returns:
            `StoryGraph` containing all loaded stories.
        """
        ...

    def get_flows(self) -> FlowsList:
        """Retrieves the flows that should be used for training.

        Default implementation returns an empty `FlowsList`. The default
        implementation is required because of backwards compatibility.

        Returns:
            `FlowsList` containing all loaded flows.
        """
        return FlowsList(underlying_flows=[])

    def get_conversation_tests(self) -> StoryGraph:
        """Retrieves end-to-end conversation stories for testing.

        Returns:
            `StoryGraph` containing all loaded stories.
        """
        return self.get_stories()

    @abstractmethod
    def get_config(self) -> Dict:
        """Retrieves the configuration that should be used for the training.

        Returns:
            The configuration as dictionary.
        """
        ...

    @abstractmethod
    def get_config_file_for_auto_config(self) -> Optional[Text]:
        """Returns config file path for auto-config only if there is a single one."""
        ...

    @abstractmethod
    def get_nlu_data(self, language: Optional[Text] = "en") -> TrainingData:
        """Retrieves the NLU training data that should be used for training.

        Args:
            language: Can be used to only load training data for a certain language.

        Returns:
            Loaded NLU `TrainingData`.
        """
        ...

    @staticmethod
    def load_from_config(
        config_path: Text,
        domain_path: Optional[Text] = None,
        training_data_paths: Optional[List[Text]] = None,
        args: Optional[Dict[Text, Any]] = None,
    ) -> "TrainingDataImporter":
        """Loads a `TrainingDataImporter` instance from a configuration file."""
        config = read_config_file(config_path)
        return TrainingDataImporter.load_from_dict(
            config, config_path, domain_path, training_data_paths, args
        )

    @staticmethod
    def load_core_importer_from_config(
        config_path: Text,
        domain_path: Optional[Text] = None,
        training_data_paths: Optional[List[Text]] = None,
        args: Optional[Dict[Text, Any]] = None,
    ) -> "TrainingDataImporter":
        """Loads core `TrainingDataImporter` instance.

        Instance loaded from configuration file will only read Core training data.
        """
        importer = TrainingDataImporter.load_from_config(
            config_path, domain_path, training_data_paths, args
        )
        return importer

    @staticmethod
    def load_nlu_importer_from_config(
        config_path: Text,
        domain_path: Optional[Text] = None,
        training_data_paths: Optional[List[Text]] = None,
        args: Optional[Dict[Text, Any]] = None,
    ) -> "TrainingDataImporter":
        """Loads nlu `TrainingDataImporter` instance.

        Instance loaded from configuration file will only read NLU training data.
        """
        importer = TrainingDataImporter.load_from_config(
            config_path, domain_path, training_data_paths, args
        )

        if isinstance(importer, E2EImporter):
            # When we only train NLU then there is no need to enrich the data with
            # E2E data from Core training data.
            importer = importer._importer

        return NluDataImporter(importer)

    @staticmethod
    def load_from_dict(
        config: Optional[Dict] = None,
        config_path: Optional[Text] = None,
        domain_path: Optional[Text] = None,
        training_data_paths: Optional[List[Text]] = None,
        args: Optional[Dict[Text, Any]] = None,
        expand_env_vars: bool = True,
    ) -> "TrainingDataImporter":
        """Loads a `TrainingDataImporter` instance from a dictionary."""
        from rasa.shared.importers.rasa import RasaFileImporter

        config = config or {}
        importers = config.get("importers", [])
        importers = [
            TrainingDataImporter._importer_from_dict(
                importer, config_path, domain_path, training_data_paths, args
            )
            for importer in importers
        ]
        importers = [importer for importer in importers if importer]
        if not importers:
            importers = [
                RasaFileImporter(
                    config_path,
                    domain_path,
                    training_data_paths,
                    expand_env_vars=expand_env_vars,
                )
            ]

        return TrainingDataImporter.wrap_in_builtins(importers)

    @staticmethod
    def wrap_in_builtins(
        importers: List["TrainingDataImporter"],
    ) -> "TrainingDataImporter":
        return LanguageImporter(
            E2EImporter(
                FlowSyncImporter(ResponsesSyncImporter(CombinedDataImporter(importers)))
            )
        )

    @staticmethod
    def _importer_from_dict(
        importer_config: Dict,
        config_path: Text,
        domain_path: Optional[Text] = None,
        training_data_paths: Optional[List[Text]] = None,
        args: Optional[Dict[Text, Any]] = None,
    ) -> Optional["TrainingDataImporter"]:
        from rasa.shared.importers.multi_project import MultiProjectImporter
        from rasa.shared.importers.rasa import RasaFileImporter
        from rasa.shared.importers.remote_importer import RemoteTrainingDataImporter

        module_path = importer_config.pop("name", None)
        if module_path == RasaFileImporter.__name__:
            importer_class: Type[TrainingDataImporter] = RasaFileImporter
        elif module_path == MultiProjectImporter.__name__:
            importer_class = MultiProjectImporter
        elif module_path == RemoteTrainingDataImporter.__name__:
            importer_class = RemoteTrainingDataImporter
        else:
            try:
                importer_class = rasa.shared.utils.common.class_from_module_path(
                    module_path
                )
            except (AttributeError, ImportError):
                logging.warning(f"Importer '{module_path}' not found.")
                return None

        constructor_arguments = rasa.shared.utils.common.minimal_kwargs(
            {**importer_config, **(args or {})}, importer_class
        )
        return importer_class(
            config_path,
            domain_path,
            training_data_paths,
            **constructor_arguments,
        )

    @staticmethod
    def fingerprint() -> Text:
        """Returns a random fingerprint as data shouldn't be cached."""
        return rasa.shared.utils.io.random_string(25)

    def __repr__(self) -> Text:
        """Returns text representation of object."""
        return self.__class__.__name__

    def get_user_flows(self) -> FlowsList:
        """Retrieves the user-defined flows that should be used for training.

        Implemented by FlowSyncImporter and E2EImporter only.

        Returns:
            `FlowsList` containing all loaded flows.
        """
        raise NotImplementedError

    def get_user_domain(self) -> Domain:
        """Retrieves the user-defined domain that should be used for training.

        Implemented by FlowSyncImporter and E2EImporter only.

        Returns:
            `Domain`.
        """
        raise NotImplementedError

    @staticmethod
    def get_domain_files(
        data_paths: Optional[Union[List[Text], Text]] = None,
    ) -> List[Text]:
        """Returns the domain file path (see parent class for full docstring)."""
        return rasa.shared.data.get_data_files(data_paths, Domain.is_domain_file)


class NluDataImporter(TrainingDataImporter):
    """Importer that skips any Core-related file reading."""

    def __init__(self, actual_importer: TrainingDataImporter):
        """Initializes the NLUDataImporter."""
        self._importer = actual_importer

    def get_domain(self) -> Domain:
        """Retrieves model domain (see parent class for full docstring)."""
        return Domain.empty()

    def get_stories(self, exclusion_percentage: Optional[int] = None) -> StoryGraph:
        """Retrieves training stories / rules (see parent class for full docstring)."""
        return StoryGraph([])

    def get_conversation_tests(self) -> StoryGraph:
        """Retrieves conversation test stories (see parent class for full docstring)."""
        return StoryGraph([])

    def get_config(self) -> Dict:
        """Retrieves model config (see parent class for full docstring)."""
        return self._importer.get_config()

    def get_nlu_data(self, language: Optional[Text] = "en") -> TrainingData:
        """Retrieves NLU training data (see parent class for full docstring)."""
        return self._importer.get_nlu_data(language)

    def get_config_file_for_auto_config(self) -> Optional[Text]:
        """Returns config file path for auto-config only if there is a single one."""
        return self._importer.get_config_file_for_auto_config()


class CombinedDataImporter(TrainingDataImporter):
    """A `TrainingDataImporter` that combines multiple importers.

    Uses multiple `TrainingDataImporter` instances
    to load the data as if they were a single instance.
    """

    def __init__(self, importers: List[TrainingDataImporter]):
        self._importers = importers

    @cached_method
    def get_config(self) -> Dict:
        """Retrieves model config (see parent class for full docstring)."""
        configs = [importer.get_config() for importer in self._importers]

        return reduce(lambda merged, other: {**merged, **(other or {})}, configs, {})

    @cached_method
    def get_domain(self) -> Domain:
        """Retrieves model domain (see parent class for full docstring)."""
        domains = [importer.get_domain() for importer in self._importers]

        return reduce(
            lambda merged, other: merged.merge(other),
            domains,
            Domain.empty(),
        )

    @cached_method
    def get_stories(self, exclusion_percentage: Optional[int] = None) -> StoryGraph:
        """Retrieves training stories / rules (see parent class for full docstring)."""
        stories = [
            importer.get_stories(exclusion_percentage) for importer in self._importers
        ]

        return reduce(
            lambda merged, other: merged.merge(other), stories, StoryGraph([])
        )

    @cached_method
    def get_flows(self) -> FlowsList:
        """Retrieves training stories / rules (see parent class for full docstring)."""
        flow_lists = [importer.get_flows() for importer in self._importers]

        return reduce(
            lambda merged, other: merged.merge(other),
            flow_lists,
            FlowsList(underlying_flows=[]),
        )

    @cached_method
    def get_conversation_tests(self) -> StoryGraph:
        """Retrieves conversation test stories (see parent class for full docstring)."""
        stories = [importer.get_conversation_tests() for importer in self._importers]

        return reduce(
            lambda merged, other: merged.merge(other), stories, StoryGraph([])
        )

    @cached_method
    def get_nlu_data(self, language: Optional[Text] = "en") -> TrainingData:
        """Retrieves NLU training data (see parent class for full docstring)."""
        nlu_data = [importer.get_nlu_data(language) for importer in self._importers]

        return reduce(
            lambda merged, other: merged.merge(other), nlu_data, TrainingData()
        )

    @cached_method
    def get_config_file_for_auto_config(self) -> Optional[Text]:
        """Returns config file path for auto-config only if there is a single one."""
        if len(self._importers) != 1:
            rasa.shared.utils.io.raise_warning(
                "Auto-config for multiple importers is not supported; "
                "using config as is."
            )
            return None
        return self._importers[0].get_config_file_for_auto_config()


class PassThroughImporter(TrainingDataImporter):
    """Importer that passes through all calls to the actual importer."""

    def __init__(self, importer: TrainingDataImporter):
        """Initializes the FlowSyncImporter."""
        self._importer = importer

    def get_config(self) -> Dict:
        """Retrieves model config (see parent class for full docstring)."""
        return self._importer.get_config()

    def get_flows(self) -> FlowsList:
        """Retrieves model flows (see parent class for full docstring)."""
        return self._importer.get_flows()

    def get_config_file_for_auto_config(self) -> Optional[Text]:
        """Returns config file path for auto-config only if there is a single one."""
        return self._importer.get_config_file_for_auto_config()

    def get_domain(self) -> Domain:
        """Retrieves model domain (see parent class for full docstring)."""
        return self._importer.get_domain()

    def get_stories(self, exclusion_percentage: Optional[int] = None) -> StoryGraph:
        """Retrieves training stories / rules (see parent class for full docstring)."""
        return self._importer.get_stories(exclusion_percentage)

    def get_conversation_tests(self) -> StoryGraph:
        """Retrieves conversation test stories (see parent class for full docstring)."""
        return self._importer.get_conversation_tests()

    def get_nlu_data(self, language: Optional[Text] = "en") -> TrainingData:
        """Updates NLU data with responses for retrieval intents from domain."""
        return self._importer.get_nlu_data(language)


DEFAULT_PATTERN_FLOWS_FILE_NAME = "default_flows_for_patterns.yml"


class FlowSyncImporter(PassThroughImporter):
    """Importer that syncs `flows` between Domain and flow training data."""

    @staticmethod
    def load_default_pattern_flows() -> FlowsList:
        """Loads the default flows from the file system."""
        from rasa.shared.core.flows.yaml_flows_io import YAMLFlowsReader

        flows = YAMLFlowsReader.read_from_file(FlowSyncImporter.default_pattern_path())
        flows.validate()
        return flows

    @staticmethod
    def default_pattern_path() -> str:
        return str(
            importlib_resources.files("rasa.dialogue_understanding.patterns").joinpath(
                DEFAULT_PATTERN_FLOWS_FILE_NAME
            )
        )

    @staticmethod
    def load_default_pattern_flows_domain() -> Domain:
        """Loads the default flows from the file system."""
        return Domain.from_path(FlowSyncImporter.default_pattern_path())

    @classmethod
    def merge_with_default_flows(cls, flows: FlowsList) -> FlowsList:
        """Merges the passed flows with the default flows.

        If a user defined flow contains a flow with an id of a default flow,
        it will overwrite the default flow.

        Args:
            flows: user defined flows.

        Returns:
        Merged flows.
        """
        default_flows = cls.load_default_pattern_flows()

        user_flow_ids = [flow.id for flow in flows.underlying_flows]
        missing_default_flows = [
            default_flow
            for default_flow in default_flows.underlying_flows
            if default_flow.id not in user_flow_ids
        ]

        return flows.merge(FlowsList(missing_default_flows))

    @cached_method
    def get_flows(self) -> FlowsList:
        flows = self._importer.get_flows()

        if flows.is_empty():
            # if there are no flows, we don't need to add the default flows either
            return flows

        return self.merge_with_default_flows(flows)

    @cached_method
    def get_user_flows(self) -> FlowsList:
        return self._importer.get_flows()

    @cached_method
    def get_domain(self) -> Domain:
        """Merge existing domain with properties of flows."""
        # load domain data from user defined domain files
        user_defined_domain = self._importer.get_domain()

        # load user defined flows
        flows = self.get_flows()
        if flows.is_empty():
            # if there are no flows, we don't need to add the default flows either
            return user_defined_domain

        default_flows_domain = self.load_default_pattern_flows_domain()

        flow_names = [
            rasa.shared.constants.FLOW_PREFIX + flow.id
            for flow in flows.underlying_flows
        ]
        flow_domain = Domain.from_dict({KEY_ACTIONS: flow_names})

        default_domain = flow_domain.merge(default_flows_domain)
        # we're merging with the default domain which contains default patterns
        # utterances (overrides)
        domain = user_defined_domain.merge(
            default_domain, ignore_warnings_about_duplicates=True
        )
        return domain

    @rasa.shared.utils.common.cached_method
    def get_user_domain(self) -> Domain:
        """Retrieves only user defined domain."""
        return self._importer.get_domain()


class LanguageImporter(PassThroughImporter):
    """Importer that configures the language settings in the domain."""

    @cached_method
    def get_domain(self) -> Domain:
        domain = self._importer.get_domain()
        if domain.is_empty():
            return domain

        domain.remove_builtin_slots()
        slot_name = rasa.shared.core.constants.LANGUAGE_SLOT
        if any(slot.name == slot_name for slot in domain.slots):
            raise RasaException(
                f"The '{slot_name}' slot is a builtin slot that cannot be overridden. "
                f"Please remove its definition from your domain configuration."
            )

        config = self._importer.get_config()
        language = config.get(CONFIG_LANGUAGE_KEY)
        additional_languages = config.get(CONFIG_ADDITIONAL_LANGUAGES_KEY) or []

        values = additional_languages.copy()
        if language and language not in values:
            values.append(language)

        # Prepare the serialized representation of the language slot
        serialized_slot: Dict[Text, Any] = {
            "type": StrictCategoricalSlot.type_name,
            "initial_value": language,
            "values": values,
            "mappings": [],
            "is_builtin": True,
        }

        domain_with_language_slot = Domain.from_dict(
            {KEY_SLOTS: {slot_name: serialized_slot}}
        )
        return domain.merge(domain_with_language_slot)

    @cached_method
    def get_user_domain(self) -> Domain:
        """Delegate to the underlying importer to get the user domain."""
        return self._importer.get_user_domain()

    @cached_method
    def get_user_flows(self) -> FlowsList:
        """Delegate to the underlying importer to get user flows."""
        return self._importer.get_user_flows()


class ResponsesSyncImporter(PassThroughImporter):
    """Importer that syncs `responses` between Domain and NLU training data.

    Synchronizes responses between Domain and NLU and
    adds retrieval intent properties from the NLU training data
    back to the Domain.
    """

    @cached_method
    def get_domain(self) -> Domain:
        """Merge existing domain with properties of retrieval intents in NLU data."""
        existing_domain = self._importer.get_domain()
        existing_nlu_data = self._importer.get_nlu_data()

        # Merge responses from NLU data with responses in the domain.
        # If NLU data has any retrieval intents, then add corresponding
        # retrieval actions with `utter_` prefix automatically to the
        # final domain, update the properties of existing retrieval intents.
        domain_with_retrieval_intents = self._get_domain_with_retrieval_intents(
            existing_nlu_data.retrieval_intents,
            existing_nlu_data.responses,
            existing_domain,
        )

        existing_domain = existing_domain.merge(
            domain_with_retrieval_intents, override=True
        )
        existing_domain.check_missing_responses()

        return existing_domain

    @staticmethod
    def _construct_retrieval_action_names(retrieval_intents: Set[Text]) -> List[Text]:
        """Lists names of all retrieval actions related to passed retrieval intents.

        Args:
            retrieval_intents: List of retrieval intents defined in the NLU training
                data.

        Returns: Names of corresponding retrieval actions
        """
        return [
            f"{rasa.shared.constants.UTTER_PREFIX}{intent}"
            for intent in retrieval_intents
        ]

    @staticmethod
    def _get_domain_with_retrieval_intents(
        retrieval_intents: Set[Text],
        responses: Dict[Text, List[Dict[Text, Any]]],
        existing_domain: Domain,
    ) -> Domain:
        """Construct a domain consisting of retrieval intents.

         The result domain will have retrieval intents that are listed
         in the NLU training data.

        Args:
            retrieval_intents: Set of retrieval intents defined in NLU training data.
            responses: Responses defined in NLU training data.
            existing_domain: Domain which is already loaded from the domain file.

        Returns: Domain with retrieval actions added to action names and properties
          for retrieval intents updated.
        """
        # Get all the properties already defined
        # for each retrieval intent in other domains
        # and add the retrieval intent property to them
        retrieval_intent_properties = []
        for intent in retrieval_intents:
            intent_properties = (
                existing_domain.intent_properties[intent]
                if intent in existing_domain.intent_properties
                else {}
            )
            intent_properties[IS_RETRIEVAL_INTENT_KEY] = True
            retrieval_intent_properties.append({intent: intent_properties})

        action_names = ResponsesSyncImporter._construct_retrieval_action_names(
            retrieval_intents
        )

        return Domain.from_dict(
            {
                KEY_INTENTS: retrieval_intent_properties,
                KEY_RESPONSES: responses,
                KEY_ACTIONS: action_names,
            }
        )

    @cached_method
    def get_nlu_data(self, language: Optional[Text] = "en") -> TrainingData:
        """Updates NLU data with responses for retrieval intents from domain."""
        existing_nlu_data = self._importer.get_nlu_data(language)
        existing_domain = self._importer.get_domain()

        return existing_nlu_data.merge(
            self._get_nlu_data_with_responses(
                existing_domain.retrieval_intent_responses
            )
        )

    @staticmethod
    def _get_nlu_data_with_responses(
        responses: Dict[Text, List[Dict[Text, Any]]],
    ) -> TrainingData:
        """Construct training data object with only the responses supplied.

        Args:
            responses: Responses the NLU data should
            be initialized with.

        Returns: TrainingData object with responses.

        """
        return TrainingData(responses=responses)


class E2EImporter(PassThroughImporter):
    """Importer with the following functionality.

    - enhances the NLU training data with actions / user messages from the stories.
    - adds potential end-to-end bot messages from stories as actions to the domain
    """

    @cached_method
    def get_user_flows(self) -> FlowsList:
        if not isinstance(self._importer, FlowSyncImporter):
            raise NotImplementedError(
                "Accessing user flows is only supported with FlowSyncImporter."
            )

        return self._importer.get_user_flows()

    @cached_method
    def get_domain(self) -> Domain:
        """Merge existing domain with properties of end-to-end actions in stories.

        Returns: Domain with end-to-end actions added to action names.
        """
        original = self._importer.get_domain()
        e2e_domain = self._get_domain_with_e2e_actions()

        return original.merge(e2e_domain)

    @rasa.shared.utils.common.cached_method
    def get_user_domain(self) -> Domain:
        """Retrieves only user defined domain."""
        if not isinstance(self._importer, FlowSyncImporter):
            raise NotImplementedError(
                "Accessing user domain is only supported with FlowSyncImporter."
            )
        return self._importer.get_user_domain()

    def _get_domain_with_e2e_actions(self) -> Domain:
        stories = self.get_stories()

        additional_e2e_action_names = set()
        for story_step in stories.story_steps:
            additional_e2e_action_names.update(
                {
                    event.action_text
                    for event in story_step.events
                    if isinstance(event, ActionExecuted) and event.action_text
                }
            )

        return Domain.from_dict({KEY_E2E_ACTIONS: list(additional_e2e_action_names)})

    @cached_method
    def get_nlu_data(self, language: Optional[Text] = "en") -> TrainingData:
        """Retrieves NLU training data (see parent class for full docstring)."""
        training_datasets = [
            _additional_training_data_from_default_actions(),
            self._importer.get_nlu_data(language),
            self._additional_training_data_from_stories(),
        ]

        return reduce(
            lambda merged, other: merged.merge(other), training_datasets, TrainingData()
        )

    def _additional_training_data_from_stories(self) -> TrainingData:
        stories = self.get_stories()

        utterances, actions = _unique_events_from_stories(stories)

        # Sort events to guarantee deterministic behavior and to avoid that the NLU
        # model has to be retrained due to changes in the event order within
        # the stories.
        sorted_utterances = sorted(
            utterances, key=lambda user: user.intent_name or user.text or ""
        )
        sorted_actions = sorted(
            actions, key=lambda action: action.action_name or action.action_text or ""
        )

        additional_messages_from_stories = [
            _messages_from_action(action) for action in sorted_actions
        ] + [_messages_from_user_utterance(user) for user in sorted_utterances]

        logger.debug(
            f"Added {len(additional_messages_from_stories)} training data examples "
            f"from the story training data."
        )
        return TrainingData(additional_messages_from_stories)


def _unique_events_from_stories(
    stories: StoryGraph,
) -> Tuple[Set[UserUttered], Set[ActionExecuted]]:
    action_events = set()
    user_events = set()

    for story_step in stories.story_steps:
        for event in story_step.events:
            if isinstance(event, ActionExecuted):
                action_events.add(event)
            elif isinstance(event, UserUttered):
                user_events.add(event)

    return user_events, action_events


def _messages_from_user_utterance(event: UserUttered) -> Message:
    # sub state correctly encodes intent vs text
    data = cast(Dict[Text, Any], event.as_sub_state())
    # sub state stores entities differently
    if data.get(ENTITIES) and event.entities:
        data[ENTITIES] = event.entities

    return Message(data=data)


def _messages_from_action(event: ActionExecuted) -> Message:
    # sub state correctly encodes action_name vs action_text
    return Message(data=event.as_sub_state())


def _additional_training_data_from_default_actions() -> TrainingData:
    additional_messages_from_default_actions = [
        Message(data={ACTION_NAME: action_name})
        for action_name in rasa.shared.core.constants.DEFAULT_ACTION_NAMES
    ]

    return TrainingData(additional_messages_from_default_actions)
