from __future__ import annotations
import os
from abc import ABC, abstractmethod
from typing import (
    Dict,
    Iterator,
    Optional,
    Set,
    Text,
    List,
    Tuple,
    Type,
    TypeVar,
    Union,
)

from pathlib import Path
from dataclasses import dataclass

import rasa.shared.core.constants
import rasa.shared.nlu.constants
import rasa.shared.utils.validation
import rasa.shared.utils.io
import rasa.shared.utils.common
from rasa.shared.data import is_likely_yaml_file
from rasa.shared.exceptions import InvalidConfigException, RasaException
from rasa.shared.core.events import ActionExecuted, UserUttered, Event

import logging

logger = logging.getLogger(__name__)


class MarkerRegistry:
    """Keeps track of tags that can be used to configure markers."""

    all_tags: Set[Text] = set()
    condition_tag_to_marker_class: Dict[Text, Type[AtomicMarker]] = {}
    operator_tag_to_marker_class: Dict[Text, Type[CompoundMarker]] = {}
    marker_class_to_tag: Dict[Type[Marker], Text] = {}
    negated_tag_to_tag: Dict[Text, Text] = {}
    tag_to_negated_tag: Dict[Text, Text] = {}

    @classmethod
    def register_builtin_markers(cls) -> None:
        """Must import all modules containing markers."""
        import rasa.core.evaluation.marker  # noqa

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

        tag = marker_class.tag()
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
        if issubclass(marker_class, AtomicMarker):
            cls.condition_tag_to_marker_class[positive_tag] = marker_class
        else:
            cls.operator_tag_to_marker_class[positive_tag] = marker_class
        cls.marker_class_to_tag[marker_class] = positive_tag


# We allow multiple atomic markers to be grouped under the same tag e.g.
# 'slot_set: ["slot_a", "slot_b"]' (see `AtomicMarkers` / `CompoundMarkers`),
# which is why this config maps to a list of texts or just one text:
ConditionConfigList = Dict[Text, Union[Text, List[Text]]]
# Compound markers can be nested:
OperatorConfig = Dict[Text, List[Union["OperatorConfig", ConditionConfigList]]]
# In case no compound operator is defined, "and" is used by default. Hence,
# a marker config can also just consist of the config for a condition:
MarkerConfig = Union[ConditionConfigList, OperatorConfig]


class InvalidMarkerConfig(RasaException):
    """Exception that can be raised when the config for a marker is not valid."""


@dataclass
class EventMetaData:
    """Describes meta data per event in some dialogue."""

    idx: int
    preceding_user_turns: int


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

    def __init__(self, name: Optional[Text] = None, negated: bool = False) -> None:
        """Instantiates a marker.

        Args:
            name: a custom name that can be used to replace the default string
                conversion of this marker
            negated: whether this marker should be negated (i.e. a negated marker
                applies if and only if the non-negated marker does not apply)
        """
        if name == Marker.ANY_MARKER:
            raise InvalidMarkerConfig(
                f"You must not use the special marker name {Marker.ANY_MARKER}. "
                f"This is to avoid confusion when you generate a marker from a "
                f"dictionary of marker configurations, which will lead to all "
                f"markers being combined under one common `ORMarker` named "
                f"{Marker.ANY_MARKER}."
            )
        self.name = name
        self.history: List[bool] = []
        self.negated: bool = negated

    def __str__(self) -> Text:
        return self.name or repr(self)

    def __repr__(self) -> Text:
        tag = str(self.negated_tag()) if self.negated else self.tag()
        return self._to_str_with(tag)

    @classmethod
    @abstractmethod
    def tag(cls) -> Text:
        """Returns the tag to be used in a config file."""
        ...

    @classmethod
    def negated_tag(cls) -> Optional[Text]:
        """Returns the tag to be used in a config file for the negated version."""
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
    def __iter__(self) -> Iterator[Marker]:
        """Returns an iterator over all markers that are part of this marker.

        Returns:
            an iterator over all markers that are part of this marker
        """
        ...

    @abstractmethod
    def __len__(self) -> int:
        """Returns the count of all markers that are part of this marker."""
        ...

    def evaluate_events(
        self, events: List[Event], recursive: bool = False
    ) -> List[Dict[Text, List[EventMetaData]]]:
        """Resets the marker, tracks all events, and collects some information.

        The collected information includes:
        - the timestamp of each event where the marker applied and
        - the number of user turns that preceded that timestamp

        If this marker is the special `ANY_MARKER` (identified by its name), then
        results will be collected for all (immediate) sub-markers.

        If `recursive` is set to `True`, then all included markers are evaluated.

        Args:
            events: a list of events describing a conversation
            recursive: set this to `True` to collect evaluations for all markers that
               this marker consists of
        Returns:
            a list that contains, for each dialogue contained in the tracker, a
            dictionary mapping that maps marker names to meta data of relevant
            events
        """
        # determine which marker to extract results from
        markers_to_be_evaluated: List[Marker] = []
        if recursive:
            markers_to_be_evaluated = [marker for marker in self]
        elif isinstance(self, CompoundMarker) and self.name == Marker.ANY_MARKER:
            markers_to_be_evaluated = self.sub_markers
        else:
            markers_to_be_evaluated = [self]

        # split the events into dialogues and evaluate them separately
        dialogues_and_start_indices = self._split_sessions(events=events)

        extracted_markers: List[Dict[Text, List[EventMetaData]]] = []
        for dialogue, start_idx in dialogues_and_start_indices:
            # track all events and collect meta data per timestep
            meta_data = self._track_all_and_collect_meta_data(
                events=dialogue, event_idx_offset=start_idx
            )
            # for each marker, keep only certain meta data
            extracted: Dict[Text, EventMetaData] = {
                str(marker): [meta_data[idx] for idx in marker.relevant_events()]
                for marker in markers_to_be_evaluated
            }
            extracted_markers.append(extracted)
        return extracted_markers

    @staticmethod
    def _split_sessions(events: List[Event]) -> List[Tuple[List[Event], int]]:
        """Identifies single dialogues in a the given sequence of events.

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
        dialogues_and_start_indices: List[Tuple[List[Event], int]] = []
        for dialogue_idx in range(len(session_start_indices)):
            start_idx = (
                session_start_indices[dialogue_idx - 1] if (dialogue_idx > 0) else 0
            )
            end_idx = session_start_indices[dialogue_idx]
            dialogue = [events[idx] for idx in range(start_idx, end_idx)]
            dialogues_and_start_indices.append((dialogue, start_idx))
        last_dialogue = [
            events[idx] for idx in range(session_start_indices[-1], len(events))
        ]
        dialogues_and_start_indices.append((last_dialogue, session_start_indices[-1]))
        return dialogues_and_start_indices

    def _track_all_and_collect_meta_data(
        self, events: List[Event], event_idx_offset: int = 0
    ) -> List[EventMetaData]:
        """Resets the marker, tracks all events, and collects metadata.

        Args:
            events: all events of a *single* dialogue that should be tracked and
                evaluated
            event_idx_offset: offset that will be used to modify the collected event
                meta data, i.e. all event indices will be shifted by this offset
        Returns:
            metadata for each tracked event with all event indices shifted by the
            given `event_idx_offset`
        """
        self.reset()
        dialogue_meta_data: List[EventMetaData] = []
        num_preceeding_user_turns = 0
        for idx, event in enumerate(events):
            is_user_turn = isinstance(event, UserUttered)
            dialogue_meta_data.append(
                EventMetaData(
                    idx=idx + event_idx_offset,
                    preceding_user_turns=num_preceeding_user_turns,
                )
            )
            self.track(event=event)
            num_preceeding_user_turns += int(is_user_turn)
        return dialogue_meta_data

    def relevant_events(self) -> List[int]:
        """Returns the indices of those tracked events that are relevant for evaluation.

        Note: Overwrite this method if you create a new marker class that should *not*
        contain meta data about each event where the marker applied in the final
        evaluation (see `evaluate_events`).

        Returns:
            indices of tracked events
        """
        return [idx for (idx, applies) in enumerate(self.history) if applies]

    @staticmethod
    def from_path(path: Union[Path, Text]) -> Marker:
        """Loads markers from one config file or all config files in a directory tree.

        For more details, see `from_config_dict`.

        Args:
            path: either the path to a single config file or the root of the directory
                tree that contains marker config files
        Returns:
            all configured markers, combined into one marker object
        """
        path = os.path.abspath(path)
        if os.path.isfile(path):
            config = rasa.shared.utils.io.read_yaml_file(path)
        elif os.path.isdir(path):
            config = Marker._load_and_combine_config_files_under(root_dir=path)
        else:
            config = {}
        if not config:
            raise InvalidMarkerConfig(f"Could not load any markers from '{path}'.")
        return Marker.from_config_dict(config)

    @staticmethod
    def _load_and_combine_config_files_under(root_dir: Text) -> MarkerConfig:
        combined_configs = {}
        yaml_files = [
            os.path.join(root, file)
            for root, _, files in os.walk(root_dir, followlinks=True)
            for file in files
            if is_likely_yaml_file(file)
        ]
        for yaml_file in yaml_files:
            config = rasa.shared.utils.io.read_yaml_file(yaml_file)
            # TODO: validation
            if set(config.keys()).intersection(combined_configs.keys()):
                raise InvalidMarkerConfig(
                    f"The names of markers defined in {yaml_file} "
                    f"({sorted(config.keys())}) "
                    f"overlap with the names of markers loaded so far "
                    f"({sorted(combined_configs.keys())})."
                )
            combined_configs.extend(config)
            logging.info(f"Added markers from {yaml_file}")
        return combined_configs

    @staticmethod
    def from_config_dict(config: Dict[Text, MarkerConfig]) -> Marker:
        """Creates markers from a dictionary of marker configurations.

        If there is more than one custom marker defined in the given dictionary,
        then the returned marker will be an `or` combination of all defined markers
        named `ANY_MARKER`.
        During evaluation, where we usually only return results for the top-level
        marker, we identify this special marker by it's name and return evaluations
        for all combined markers instead.

        Args:
            config: mapping custom marker names to marker configurations
        Returns:
            all configured markers, combined into one marker
        """
        from rasa.core.evaluation.marker import OrMarker

        markers = [
            Marker.from_config(marker_config, name=marker_name)
            for marker_name, marker_config in config.items()
        ]
        if len(markers) > 1:
            marker = OrMarker(markers=markers)
            marker.name = Marker.ANY_MARKER  # cannot be set via name parameter
        else:
            marker = markers[0]
        return marker

    @staticmethod
    def from_config(config: MarkerConfig, name: Optional[Text] = None) -> Marker:
        """Creates a marker from the given config.

        Args:
            config: the configuration of a single or multiple markers
            name: a custom name that will be used for the top-level marker (if and
                only if there is only one top-level marker)

        Returns:
            the configured marker
        """
        # Triggers the import of all modules containing marker classes in order to
        # register all configurable markers.
        MarkerRegistry.register_builtin_markers()
        from rasa.core.evaluation.marker import AndMarker

        # A marker config can be either an atomic marker config list or a
        # compound marker configuration - if it is not a list, we
        marker_configs = config if isinstance(config, list) else [config]

        collected_sub_markers = []
        for marker_config in marker_configs:

            # FIXME: check that this is basically a tuple
            marker_name = next(iter(marker_config))
            sub_marker_config = marker_config[marker_name]

            tag, _ = MarkerRegistry.get_non_negated_tag(tag_or_negated_tag=marker_name)
            if tag in MarkerRegistry.operator_tag_to_marker_class:
                sub_markers = [
                    CompoundMarker.from_config(
                        operator=marker_name,
                        sub_marker_configs=sub_marker_config,
                        name=name,
                    )
                ]
            else:
                sub_markers = AtomicMarker.from_config(
                    marker_name=marker_name,
                    sub_marker_config=sub_marker_config,
                    name=name,
                )
            collected_sub_markers.extend(sub_markers)

        # Build the final marker
        if len(collected_sub_markers) > 1:
            marker = AndMarker(collected_sub_markers, name=name)
        else:
            marker = collected_sub_markers[0]
            marker.name = name
        return marker


class CompoundMarker(Marker, ABC):
    """Combines several markers into one."""

    def __init__(
        self, markers: List[Marker], negated: bool = False, name: Optional[Text] = None
    ) -> None:
        """Instantiates a marker.

        Args:
            markers: the list of markers to combine
            negated: whether this marker should be negated (i.e. a negated marker
                applies if and only if the non-negated marker does not apply)
            name: a custom name that can be used to replace the default string
                conversion of this marker
        """
        super().__init__(name=name, negated=negated)
        self.sub_markers: List[Marker] = markers

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

    def __iter__(self) -> Iterator[Marker]:
        """Returns an iterator over all included markers, plus this marker itself.

        Returns:
            an iterator over all markers that are part of this marker
        """
        for marker in self.sub_markers:
            for sub_marker in marker:
                yield sub_marker
        yield self

    def __len__(self) -> int:
        """Returns the count of all markers that are part of this marker."""
        return len(self.sub_markers) + 1

    def reset(self) -> None:
        """Evaluate this marker given the next event.

        Args:
            event: the next event of the conversation
        """
        for marker in self.sub_markers:
            marker.reset()
        super().reset()

    @staticmethod
    def from_config(
        operator: Text,
        sub_marker_configs: List[Union[ConditionConfigList, OperatorConfig]],
        name: Optional[Text] = None,
    ) -> Marker:
        """Creates a compound marker from the given config.

        Args:
            operator: a text identifying a compound marker type
            sub_marker_configs: a list of configs defining sub-markers
            name: an optional custom name to be attached to the resulting marker
        Returns:
           a compound marker if there are markers to combine - and just an event
           marker if there is only a single marker
        """
        tag, is_negation = MarkerRegistry.get_non_negated_tag(operator)
        operator_class = MarkerRegistry.operator_tag_to_marker_class.get(tag)
        if operator_class is None:
            raise InvalidConfigException(f"Unknown operator '{operator}'.")

        collected_sub_markers: List[Marker] = []
        for sub_marker_name_and_config in sub_marker_configs:
            sub_marker_name = next(iter(sub_marker_name_and_config))
            sub_marker_config = sub_marker_name_and_config[sub_marker_name]

            next_sub_markers: List[Marker] = []
            sub_tag, _ = MarkerRegistry.get_non_negated_tag(
                tag_or_negated_tag=sub_marker_name
            )
            if sub_tag in MarkerRegistry.condition_tag_to_marker_class:
                # We allow several AtomicMarkers to be collected under the same tag.
                # If this is done at the top-level (see `Marker.from_config`), then
                # we would combine them under a new `AndMarker`.
                # Here, we do *not* create a new `AndMarker` but add the single
                # atomic markers to this compound marker.
                next_sub_markers = AtomicMarker.from_config(
                    marker_name=sub_marker_name, sub_marker_config=sub_marker_config
                )
            else:
                next_sub_markers = [
                    CompoundMarker.from_config(
                        operator=sub_marker_name, sub_marker_configs=sub_marker_config
                    )
                ]
            collected_sub_markers.extend(next_sub_markers)

        marker = operator_class(markers=collected_sub_markers, negated=is_negation)
        marker.name = name
        return marker


class AtomicMarker(Marker, ABC):
    """A marker that does not contain any sub-markers."""

    def __init__(
        self, text: Text, negated: bool = False, name: Optional[Text] = None
    ) -> None:
        """Instantiates an atomic marker.

        Args:
            text: some text used to decide whether the marker applies
            negated: whether this marker should be negated (i.e. a negated marker
                applies if and only if the non-negated marker does not apply)
            name: a custom name that can be used to replace the default string
                conversion of this marker
        """
        super().__init__(name=name, negated=negated)
        self.text = text

    def _to_str_with(self, tag: Text) -> Text:
        return f"({tag}: {self.text})"

    def __iter__(self) -> Iterator[AtomicMarker]:
        """Returns an iterator that just returns this `AtomicMarker`.

        Returns:
            an iterator over all markers that are part of this marker, i.e. this marker
        """
        yield self

    def __len__(self) -> int:
        """Returns the count of all markers that are part of this marker."""
        return 1

    @staticmethod
    def from_config(
        marker_name: Text,
        sub_marker_config: Union[Text, List[Text]],
        name: Optional[Text] = None,
    ) -> List[AtomicMarker]:
        """Creates an atomic marker from the given config.

        Args:
            marker_name: string identifying an atomic marker type
            sub_marker_config: a list of texts or just one text which should be
               used to instantiate the condition marker(s)
            name: a custom name for this marker
        Returns:
            the configured `AtomicMarker`s
        """
        tag, is_negation = MarkerRegistry.get_non_negated_tag(marker_name)
        marker_class = MarkerRegistry.condition_tag_to_marker_class.get(tag)
        if marker_class is None:
            raise InvalidConfigException(f"Unknown condition '{marker_name}'.")

        if isinstance(sub_marker_config, Text):
            sub_marker_config_list = [sub_marker_config]
        else:
            sub_marker_config_list = sub_marker_config

        markers = []
        for marker_text in sub_marker_config_list:
            marker = marker_class(marker_text, negated=is_negation)
            marker.name = name
            markers.append(marker)
        return markers
