from __future__ import annotations

import csv
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    ClassVar,
    Dict,
    Iterator,
    List,
    Optional,
    Set,
    Text,
    Tuple,
    Type,
    TypeVar,
    Union,
)

import rasa.shared.core.constants
import rasa.shared.nlu.constants
import rasa.shared.utils.common
import rasa.shared.utils.io
from rasa.shared.constants import DOCS_URL_MARKERS
from rasa.shared.core.domain import Domain
from rasa.shared.core.events import ActionExecuted, Event, UserUttered
from rasa.shared.exceptions import InvalidConfigException, RasaException
from rasa.utils.io import WriteRow
from rasa.utils.yaml import (
    collect_configs_from_yaml_files,
    collect_yaml_files_from_path,
)

if TYPE_CHECKING:
    from rasa.markers.marker import OrMarker

logger = logging.getLogger(__name__)


class MarkerRegistry:
    """Keeps track of tags that can be used to configure markers."""

    all_tags: ClassVar[Set[Text]] = set()
    condition_tag_to_marker_class: ClassVar[Dict[Text, Type[ConditionMarker]]] = {}
    operator_tag_to_marker_class: ClassVar[Dict[Text, Type[OperatorMarker]]] = {}
    marker_class_to_tag: ClassVar[Dict[Type[Marker], Text]] = {}
    negated_tag_to_tag: ClassVar[Dict[Text, Text]] = {}
    tag_to_negated_tag: ClassVar[Dict[Text, Text]] = {}

    @classmethod
    def register_builtin_markers(cls) -> None:
        """Must import all modules containing markers."""
        import rasa.markers.marker  # noqa: F401

    @classmethod
    def configurable_marker(cls, marker_class: Type[Marker]) -> Type[Marker]:
        """Decorator used to register a marker that can be used in config files.

        Args:
            marker_class: the marker class to be made available via config files
        Returns:
            the registered marker class
        """
        if not issubclass(marker_class, Marker):
            raise RuntimeError("Can only register marker classes as configurable.")

        tag = marker_class.positive_tag()
        negated_tag = marker_class.negated_tag()
        cls._register_tags(tags={tag, negated_tag})
        cls._register_tag_class(marker_class=marker_class, positive_tag=tag)
        cls._register_negation(tag=tag, negated_tag=negated_tag)
        return marker_class

    @classmethod
    def _register_tags(cls, tags: Set[Optional[Text]]) -> None:
        specified_tags = {tag for tag in tags if tag is not None}
        for tag_ in specified_tags:
            if tag_ in cls.all_tags:
                raise RuntimeError(
                    "Expected the tags of all configurable markers to be "
                    "identifiable by their tag."
                )
            cls.all_tags.add(tag_)

    @classmethod
    def _register_negation(cls, tag: Text, negated_tag: Optional[Text]) -> None:
        if negated_tag is not None:
            cls.negated_tag_to_tag[negated_tag] = tag
            cls.tag_to_negated_tag[tag] = negated_tag

    @classmethod
    def get_non_negated_tag(cls, tag_or_negated_tag: Text) -> Tuple[Text, bool]:
        """Returns the non-negated marker tag, given a (possible) negated marker tag.

        Args:
            tag_or_negated_tag: the tag for a possibly negated marker
        Returns:
            the tag itself if it was already positive, otherwise the positive version;
            and a boolean that represents whether the given tag was a negative one
        """
        # either the given `marker_name` is already "positive" (e.g. "slot was set")
        # or there is an entry mapping it to it's positive version:
        positive_version = cls.negated_tag_to_tag.get(
            tag_or_negated_tag, tag_or_negated_tag
        )
        is_negation = tag_or_negated_tag != positive_version
        return positive_version, is_negation

    @classmethod
    def _register_tag_class(
        cls, marker_class: Type[Marker], positive_tag: Text
    ) -> None:
        if issubclass(marker_class, ConditionMarker):
            cls.condition_tag_to_marker_class[positive_tag] = marker_class
        elif issubclass(marker_class, OperatorMarker):
            cls.operator_tag_to_marker_class[positive_tag] = marker_class
        else:
            raise RuntimeError(
                f"Can only register `{OperatorMarker.__name__} or "
                f" {ConditionMarker.__name__} subclasses."
            )
        cls.marker_class_to_tag[marker_class] = positive_tag


class InvalidMarkerConfig(RasaException):
    """Exception that can be raised when the config for a marker is not valid."""


@dataclass
class EventMetaData:
    """Describes meta data per event in some session."""

    idx: int
    preceding_user_turns: int


# We evaluate markers separately against every session and extract, for every marker
# that we want to evaluate, the meta data of the respective relevant events where the
# marker applies.
SessionEvaluation = Dict[Text, List[EventMetaData]]

T = TypeVar("T")


class Marker(ABC):
    """A marker is a way of describing points in conversations you're interested in.

    Here, markers are stateful objects because they track the events of a conversation.
    At each point in the conversation, one can observe whether a marker applies or
    does not apply to the conversation so far.
    """

    # Identifier for an artificial marker that is created when loading markers
    # from a dictionary of configs. For more details, see `from_config_dict`.
    ANY_MARKER = "<any_marker>"

    def __init__(
        self,
        name: Optional[Text] = None,
        negated: bool = False,
        description: Optional[Text] = None,
    ) -> None:
        """Instantiates a marker.

        Args:
            name: a custom name that can be used to replace the default string
                conversion of this marker
            negated: whether this marker should be negated (i.e. a negated marker
                applies if and only if the non-negated marker does not apply)
            description: an optional description of the marker. It is not used
                internally but can be used to document the marker.

        Raises:
            `InvalidMarkerConfig` if the chosen *name* of the marker is the tag of
            a predefined marker.
        """
        if name == Marker.ANY_MARKER or name in MarkerRegistry.all_tags:
            raise InvalidMarkerConfig(
                f"You must not use the special marker name {Marker.ANY_MARKER}. "
                f"This is to avoid confusion when you generate a marker from a "
                f"dictionary of marker configurations, which will lead to all "
                f"markers being combined under one common `ORMarker` named "
                f"{Marker.ANY_MARKER}."
            )
        self.name = name
        self.history: List[bool] = []
        # Note: we allow negation here even though there might not be a negated tag
        # for 2 reasons: testing and the fact that the `MarkerRegistry`+`from_config`
        # won't allow to create a negated marker if there is no negated tag.
        self.negated: bool = negated
        self.description = description

    def __str__(self) -> Text:
        return self.name or repr(self)

    def __repr__(self) -> Text:
        return self._to_str_with(self.get_tag())

    def get_tag(self) -> Text:
        """Returns the tag describing this marker."""
        if self.negated:
            tag = self.negated_tag()
            if tag is None:
                # We allow the instantiation of a negated marker even if there
                # is no negated tag (ie. direct creation of the negated marker from
                # a config is not allowed). To be able to print a tag nonetheless:
                tag = f"not({self.positive_tag()})"
            return tag
        else:
            return self.positive_tag()

    @staticmethod
    @abstractmethod
    def positive_tag() -> Text:
        """Returns the tag to be used in a config file."""
        ...

    @staticmethod
    def negated_tag() -> Optional[Text]:
        """Returns the tag to be used in a config file for the negated version.

        If this maps to `None`, then this indicates that we do not allow a short-cut
        for negating this marker. Hence, there is not a single tag to instantiate
        a negated version of this marker. One must use a "not" in the configuration
        file then.
        """
        return None

    @abstractmethod
    def _to_str_with(self, tag: Text) -> Text:
        """Returns a string representation using the given tag."""
        ...

    def track(self, event: Event) -> None:
        """Updates the marker according to the given event.

        Args:
            event: the next event of the conversation
        """
        result = self._non_negated_version_applies_at(event)
        if self.negated:
            result = not result
        self.history.append(result)

    @abstractmethod
    def _non_negated_version_applies_at(self, event: Event) -> bool:
        """Checks whether the non-negated version applies at the next given event.

        This method must *not* update the marker.

        Args:
            event: the next event of the conversation
        """
        ...

    def reset(self) -> None:
        """Clears the history of the marker."""
        self.history = []

    @abstractmethod
    def flatten(self) -> Iterator[Marker]:
        """Returns an iterator over all conditions and operators used in this marker.

        Returns:
            an iterator over all conditions and operators that are part of this marker
        """
        ...

    @abstractmethod
    def validate_against_domain(self, domain: Domain) -> bool:
        """Checks that this marker (and its children) refer to entries in the domain.

        Args:
            domain: The domain to check against
        """
        ...

    @abstractmethod
    def max_depth(self) -> int:
        """Gets the maximum depth from this point in the marker tree."""
        ...

    def evaluate_events(self, events: List[Event]) -> List[SessionEvaluation]:
        """Resets the marker, tracks all events, and collects some information.

        The collected information includes:
        - the timestamp of each event where the marker applied and
        - the number of user turns that preceded that timestamp

        If this marker is the special `ANY_MARKER` (identified by its name), then
        results will be collected for all (immediate) sub-markers.

        Args:
            events: a list of events describing a conversation
        Returns:
            a list that contains, for each session contained in the tracker, a
            dictionary mapping that maps marker names to meta data of relevant
            events
        """
        # determine which marker to extract results from
        if isinstance(self, OperatorMarker) and self.name == Marker.ANY_MARKER:
            markers_to_be_evaluated = self.sub_markers
        else:
            markers_to_be_evaluated = [self]

        # split the events into sessions and evaluate them separately
        sessions_and_start_indices = self._split_sessions(events=events)

        extracted_markers: List[Dict[Text, List[EventMetaData]]] = []
        for session, start_idx in sessions_and_start_indices:
            # track all events and collect meta data per time step
            meta_data = self._track_all_and_collect_meta_data(
                events=session, event_idx_offset=start_idx
            )
            # for each marker, keep only certain meta data
            extracted: Dict[Text, List[EventMetaData]] = {
                str(marker): [meta_data[idx] for idx in marker.relevant_events()]
                for marker in markers_to_be_evaluated
            }
            extracted_markers.append(extracted)
        return extracted_markers

    @staticmethod
    def _split_sessions(events: List[Event]) -> List[Tuple[List[Event], int]]:
        """Identifies single sessions in a the given sequence of events.

        Args:
            events: a sequence of events, e.g. extracted from a tracker store
        Returns:
            a list of sub-sequences of the given events that describe single
            conversations and the respective index that describes where the
            subsequence starts in the original sequence
        """
        session_start_indices = [
            idx
            for idx, event in enumerate(events)
            if isinstance(event, ActionExecuted)
            and event.action_name
            == rasa.shared.core.constants.ACTION_SESSION_START_NAME
        ]
        if len(session_start_indices) == 0:
            return [(events, 0)]
        sessions_and_start_indices: List[Tuple[List[Event], int]] = []
        for session_idx in range(len(session_start_indices) - 1):
            start_idx = session_start_indices[session_idx]
            end_idx = session_start_indices[session_idx + 1]
            session = [events[idx] for idx in range(start_idx, end_idx)]
            sessions_and_start_indices.append((session, start_idx))
        last_session = [
            events[idx] for idx in range(session_start_indices[-1], len(events))
        ]
        sessions_and_start_indices.append((last_session, session_start_indices[-1]))
        return sessions_and_start_indices

    def _track_all_and_collect_meta_data(
        self, events: List[Event], event_idx_offset: int = 0
    ) -> List[EventMetaData]:
        """Resets the marker, tracks all events, and collects metadata.

        Args:
            events: all events of a *single* session that should be tracked and
                evaluated
            event_idx_offset: offset that will be used to modify the collected event
                meta data, i.e. all event indices will be shifted by this offset
        Returns:
            metadata for each tracked event with all event indices shifted by the
            given `event_idx_offset`
        """
        self.reset()
        session_meta_data: List[EventMetaData] = []
        num_preceding_user_turns = 0
        for idx, event in enumerate(events):
            is_user_turn = isinstance(event, UserUttered)
            session_meta_data.append(
                EventMetaData(
                    idx=idx + event_idx_offset,
                    preceding_user_turns=num_preceding_user_turns,
                )
            )
            self.track(event=event)
            num_preceding_user_turns += int(is_user_turn)
        return session_meta_data

    def relevant_events(self) -> List[int]:
        """Returns the indices of those tracked events that are relevant for evaluation.

        Note: Overwrite this method if you create a new marker class that should *not*
        contain meta data about each event where the marker applied in the final
        evaluation (see `evaluate_events`).

        Returns:
            indices of tracked events
        """
        return [idx for (idx, applies) in enumerate(self.history) if applies]

    @classmethod
    def from_path(cls, path: Union[Path, Text]) -> "OrMarker":
        """Loads markers from one config file or all config files in a directory tree.

        Each config file should contain a dictionary mapping marker names to the
        respective marker configuration.
        To avoid confusion, the marker names must not coincide with the tag of
        any pre-defined markers. Moreover, marker names must be unique. This means,
        if you you load the markers from multiple files, then you have to make sure
        that the names of the markers defined in these files are unique across all
        loaded files.

        Note that all loaded markers will be combined into one marker via one
        artificial OR-operator. When evaluating the resulting marker, then the
        artificial OR-operator will be ignored and results will be reported for
        every sub-marker.

        For more details how a single marker configuration looks like, have a look
        at `Marker.from_config`.

        Args:
            path: either the path to a single config file or the root of the directory
                tree that contains marker config files
        Returns:
            all configured markers, combined into one marker object
        """
        MarkerRegistry.register_builtin_markers()
        from rasa.markers.marker import OrMarker

        # collect all the files from which we want to load configs (i.e. either just
        # the given path if points to a yaml or all yaml-files in the directory tree)
        yaml_files = cls._collect_yaml_files_from_path(path)

        # collect all the configs from those yaml files (assure it's dictionaries
        # mapping marker names to something) -- keep track of which config came
        # from which file to be able to raise meaningful errors later
        loaded_configs: Dict[Text, Dict] = cls._collect_configs_from_yaml_files(
            yaml_files
        )

        # create markers from all loaded configurations
        loaded_markers: List[Marker] = []
        for yaml_file, config in loaded_configs.items():
            for marker_name, marker_config in config.items():
                try:
                    marker = Marker.from_config(marker_config, name=marker_name)
                except InvalidMarkerConfig as e:
                    # we don't re-raise here because the stack trace would only be
                    # printed when we run rasa evaluate with --debug flag
                    raise InvalidMarkerConfig(
                        f"Could not load marker {marker_name} from {yaml_file}. "
                        f"Reason: {e!s}. "
                    )
                loaded_markers.append(marker)

        # Reminder: We could also just create a dictionary of markers from this.
        # However, if we want to allow re-using top-level markers (e.g.
        # "custom_marker1 or custom_marker2" and/or optimize the marker evaluation such
        # that e.g. the same condition is not instantiated (and evaluated) twice, then
        # the current approach might be better (e.g. with a dictionary of markers one
        # might expect the markers to be independent objects).

        # combine the markers
        marker = OrMarker(markers=loaded_markers)
        marker.name = Marker.ANY_MARKER  # cannot be set via name parameter
        return marker

    @staticmethod
    def _collect_yaml_files_from_path(path: Union[Text, Path]) -> List[Text]:
        return collect_yaml_files_from_path(path)

    @staticmethod
    def _collect_configs_from_yaml_files(yaml_files: List[Text]) -> Dict[Text, Dict]:
        loaded_configs: Dict[Text, Dict] = collect_configs_from_yaml_files(yaml_files)

        marker_names: Set[Text] = set()
        for yaml_file, loaded_config in loaded_configs.items():
            if set(loaded_config.keys()).intersection(marker_names):
                raise InvalidMarkerConfig(
                    f"The names of markers defined in {yaml_file} "
                    f"({sorted(loaded_config.keys())}) "
                    f"overlap with the names of markers loaded so far "
                    f"({sorted(marker_names)}). "
                    f"Please adapt your configurations such that your custom "
                    f"marker names are unique. "
                    f"Refer to the docs for more information: {DOCS_URL_MARKERS} "
                )
            if set(loaded_config.keys()).intersection(MarkerRegistry.all_tags):
                raise InvalidMarkerConfig(
                    f"The top level of your marker configuration should consist "
                    f"of names for your custom markers. "
                    f"If you use a condition or operator name at the top level, "
                    f"then that will not be recognised as an actual condition "
                    f"or operator and won't be evaluated against any sessions. "
                    f"Please remove any of the pre-defined marker tags "
                    f"(i.e. {MarkerRegistry.all_tags}) "
                    f"from {yaml_file}. "
                    f"Refer to the docs for more information: {DOCS_URL_MARKERS} "
                )
            marker_names.update(loaded_config.keys())
        return loaded_configs

    @staticmethod
    def from_config(config: Any, name: Optional[Text] = None) -> Marker:
        """Creates a marker from the given config.

        A marker configuration is a dictionary mapping a marker tag (either a
        `positive_tag` or a `negated_tag`) to a sub-configuration.
        How that sub-configuration looks like, depends on whether the tag describes
        an operator (see `OperatorMarker.from_tag_and_sub_config`) or a
        condition (see `ConditionMarker.from_tag_and_sub_config`).

        Args:
            config: a marker configuration
            name: a custom name that will be used for the top-level marker (if and
                only if there is only one top-level marker)

        Returns:
            the configured marker
        """
        # Triggers the import of all modules containing marker classes in order to
        # register all configurable markers.
        MarkerRegistry.register_builtin_markers()

        if not isinstance(config, dict) or len(config) not in [1, 2]:
            raise InvalidMarkerConfig(
                "To configure a marker, please define a dictionary that maps a "
                "single operator tag or a single condition tag to the "
                "corresponding parameter configuration or a list of marker "
                "configurations, respectively. "
                f"Refer to the docs for more information: {DOCS_URL_MARKERS} "
            )

        description = config.pop("description", None)
        tag = next(iter(config))
        sub_marker_config = config[tag]

        tag, _ = MarkerRegistry.get_non_negated_tag(tag_or_negated_tag=tag)
        if tag in MarkerRegistry.operator_tag_to_marker_class:
            return OperatorMarker.from_tag_and_sub_config(
                tag=tag,
                sub_config=sub_marker_config,
                name=name,
                description=description,
            )
        elif tag in MarkerRegistry.condition_tag_to_marker_class:
            return ConditionMarker.from_tag_and_sub_config(
                tag=tag,
                sub_config=sub_marker_config,
                name=name,
                description=description,
            )

        raise InvalidMarkerConfig(
            f"Expected a marker configuration with a key that specifies"
            f" an operator or a condition but found {tag}. "
            f"Available conditions and operators are: "
            f"{sorted(MarkerRegistry.all_tags)}. "
            f"Refer to the docs for more information: {DOCS_URL_MARKERS} "
        )

    @staticmethod
    def _save_results(path: Path, results: Dict[Text, List[SessionEvaluation]]) -> None:
        """Save extracted marker results as CSV to specified path.

        Args:
            path: Path to write out the extracted markers.
            results: Extracted markers from a selection of trackers.
        """
        with path.open(mode="w") as f:
            table_writer = csv.writer(f)
            table_writer.writerow(
                [
                    "sender_id",
                    "session_idx",
                    "marker",
                    "event_idx",
                    "num_preceding_user_turns",
                ]
            )
            for sender_id, results_per_session in results.items():
                for session_idx, session_result in enumerate(results_per_session):
                    Marker._write_relevant_events(
                        table_writer, sender_id, session_idx, session_result
                    )

    @staticmethod
    def _write_relevant_events(
        writer: WriteRow, sender_id: Text, session_idx: int, session: SessionEvaluation
    ) -> None:
        for marker_name, meta_data_per_relevant_event in session.items():
            for event_meta_data in meta_data_per_relevant_event:
                writer.writerow(
                    [
                        sender_id,
                        str(session_idx),
                        marker_name,
                        str(event_meta_data.idx),
                        str(event_meta_data.preceding_user_turns),
                    ]
                )


class OperatorMarker(Marker, ABC):
    """Combines several markers into one."""

    def __init__(
        self,
        markers: List[Marker],
        negated: bool = False,
        name: Optional[Text] = None,
        description: Optional[Text] = None,
    ) -> None:
        """Instantiates a marker.

        Args:
            markers: the list of markers to combine
            negated: whether this marker should be negated (i.e. a negated marker
                applies if and only if the non-negated marker does not apply)
            name: a custom name that can be used to replace the default string
                conversion of this marker
            description: an optional description of the marker. It is not used
                internally but can be used to document the marker.

        Raises:
            `InvalidMarkerConfig` if the given number of sub-markers does not match
            the expected number of sub-markers
        """
        super().__init__(name=name, negated=negated, description=description)
        self.sub_markers: List[Marker] = markers
        expected_num = self.expected_number_of_sub_markers()
        if expected_num is not None and len(markers) != expected_num:
            raise InvalidMarkerConfig(
                f"Expected {expected_num} sub-marker(s) to be specified for marker "
                f"'{self}' ({self.get_tag()}) but found {len(markers)}. "
                f"Please adapt your configuration so that there are exactly "
                f"{expected_num} sub-markers. "
            )
        elif len(markers) == 0:
            raise InvalidMarkerConfig(
                f"Expected some sub-markers to be specified for {self} but "
                f"found none. Please adapt your configuration so that there is "
                f"at least one sub-marker. "
            )

    @staticmethod
    def expected_number_of_sub_markers() -> Optional[int]:
        """Returns the expected number of sub-markers (if there is any)."""
        return None

    def _to_str_with(self, tag: Text) -> Text:
        marker_str = ", ".join(str(marker) for marker in self.sub_markers)
        return f"{tag}({marker_str})"

    def track(self, event: Event) -> None:
        """Updates the marker according to the given event.

        All sub-markers will be updated before the compound marker itself is updated.

        Args:
            event: the next event of the conversation
        """
        for marker in self.sub_markers:
            marker.track(event)
        super().track(event)

    def flatten(self) -> Iterator[Marker]:
        """Returns an iterator over all included markers, plus this marker itself.

        Returns:
            an iterator over all markers that are part of this marker
        """
        for marker in self.sub_markers:
            for sub_marker in marker.flatten():
                yield sub_marker
        yield self

    def reset(self) -> None:
        """Resets the history of this marker and all its sub-markers."""
        for marker in self.sub_markers:
            marker.reset()
        super().reset()

    def validate_against_domain(self, domain: Domain) -> bool:
        """Checks that this marker (and its children) refer to entries in the domain.

        Args:
            domain: The domain to check against
        """
        return all(
            marker.validate_against_domain(domain) for marker in self.sub_markers
        )

    def max_depth(self) -> int:
        """Gets the maximum depth from this point in the marker tree."""
        return 1 + max(child.max_depth() for child in self.sub_markers)

    @staticmethod
    def from_tag_and_sub_config(
        tag: Text,
        sub_config: Any,
        name: Optional[Text] = None,
        description: Optional[Text] = None,
    ) -> OperatorMarker:
        """Creates an operator marker from the given config.

        The configuration must consist of a list of marker configurations.
        See `Marker.from_config` for more details.

        Args:
            tag: the tag identifying an operator
            sub_config: a list of marker configs
            name: an optional custom name to be attached to the resulting marker
            description: an optional description of the marker
        Returns:
           the configured operator marker
        Raises:
            `InvalidMarkerConfig` if the given config or the tag are not well-defined
        """
        positive_tag, is_negation = MarkerRegistry.get_non_negated_tag(tag)
        operator_class = MarkerRegistry.operator_tag_to_marker_class.get(positive_tag)
        if operator_class is None:
            raise InvalidConfigException(f"Unknown operator '{tag}'.")

        if not isinstance(sub_config, list):
            raise InvalidMarkerConfig(
                f"Expected a list of sub-marker configurations under {tag}."
            )
        collected_sub_markers: List[Marker] = []
        for sub_marker_config in sub_config:
            try:
                sub_marker = Marker.from_config(sub_marker_config)
            except InvalidMarkerConfig as e:
                # we don't re-raise here because the stack trace would only be
                # printed when we run rasa evaluate with --debug flag
                raise InvalidMarkerConfig(
                    f"Could not create sub-marker for operator '{tag}' from "
                    f"{sub_marker_config}. Reason: {e!s}"
                )
            collected_sub_markers.append(sub_marker)
        try:
            marker = operator_class(markers=collected_sub_markers, negated=is_negation)
        except InvalidMarkerConfig as e:
            # we don't re-raise here because the stack trace would only be
            # printed when we run rasa evaluate with --debug flag
            raise InvalidMarkerConfig(
                f"Could not create operator '{tag}' with sub-markers "
                f"{collected_sub_markers}. Reason: {e!s}"
            )
        marker.name = name
        marker.description = description
        return marker


class ConditionMarker(Marker, ABC):
    """A marker that does not contain any sub-markers."""

    def __init__(
        self,
        text: Text,
        negated: bool = False,
        name: Optional[Text] = None,
        description: Optional[Text] = None,
    ) -> None:
        """Instantiates an atomic marker.

        Args:
            text: some text used to decide whether the marker applies
            negated: whether this marker should be negated (i.e. a negated marker
                applies if and only if the non-negated marker does not apply)
            name: a custom name that can be used to replace the default string
                conversion of this marker
            description: an optional description of the marker. It is not used
                internally but can be used to document the marker.
        """
        super().__init__(name=name, negated=negated)
        self.text = text
        self.description = description

    def _to_str_with(self, tag: Text) -> Text:
        return f"({tag}: {self.text})"

    def flatten(self) -> Iterator[ConditionMarker]:
        """Returns an iterator that just returns this `AtomicMarker`.

        Returns:
            an iterator over all markers that are part of this marker, i.e. this marker
        """
        yield self

    def max_depth(self) -> int:
        """Gets the maximum depth from this point in the marker tree."""
        return 1

    @staticmethod
    def from_tag_and_sub_config(
        tag: Text,
        sub_config: Any,
        name: Optional[Text] = None,
        description: Optional[Text] = None,
    ) -> ConditionMarker:
        """Creates an atomic marker from the given config.

        Args:
            tag: the tag identifying a condition
            sub_config: a single text parameter expected by all condition markers;
               e.g. if the tag is for an `intent_detected` marker then the `config`
               should contain an intent's name
            name: a custom name for this marker
            description: an optional description of the marker
        Returns:
            the configured `ConditionMarker`
        Raises:
            `InvalidMarkerConfig` if the given config or the tag are not well-defined
        """
        positive_tag, is_negation = MarkerRegistry.get_non_negated_tag(tag)
        marker_class = MarkerRegistry.condition_tag_to_marker_class.get(positive_tag)
        if marker_class is None:
            raise InvalidConfigException(f"Unknown condition '{tag}'.")

        if not isinstance(sub_config, str):
            raise InvalidMarkerConfig(
                f"Expected a text parameter to be specified for marker '{tag}'."
            )
        marker = marker_class(sub_config, negated=is_negation)
        marker.name = name
        marker.description = description
        return marker
