from __future__ import annotations
import os
from abc import ABC, abstractmethod
from typing import (
    Callable,
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

from typing_extensions import TypedDict
from pathlib import Path


import rasa.shared.core.constants
import rasa.shared.nlu.constants
import rasa.shared.utils.validation
import rasa.shared.utils.io
import rasa.shared.utils.common
from rasa.shared.data import is_likely_yaml_file
from rasa.shared.exceptions import InvalidConfigException, RasaException
from rasa.shared.core.events import UserUttered, Event

import logging

logger = logging.getLogger(__name__)

# Special name of an `OrMarker` that won't be evaluated.
ANY_MARKER = "<any_marker>"

# all tags can be used in config files
TAGS: Set[Text] = set()
# mapping of all possible tags to the respective classes
TAG_TO_MARKER_CLASS: Dict[Text, Type[Marker]] = {}
MARKER_CLASS_TO_TAG: Dict[Type[Marker], Text] = {}
# mapping negated tas (e.g. 'slot_was_not_set') to regular tags ('slot_was_set')
NEGATED_TAG_TO_TAG: Dict[Text, Text] = {}
TAG_TO_NEGATED_TAG: Dict[Text, Text] = {}
# all tags that can be used in config files for a condition
CONDITION_TAGS: Set[Text] = set()
# all tags that can be used to configure operators
OPERATOR_TAGS: Set[Text] = set()


def configurable_via(
    tag: Text, negated_tag: Optional[Text] = None
) -> Callable[[Type[Marker]], Type[Marker]]:
    """Registers a marker that can be used in config files.

    Args:
        tag: the string to be used in the config file to invoke this marker
        negated_tag: the sting to be used in the config file to invoke the negated
           version of this marker
    Returns:
        a decorator
    """

    def inner(marker_class: Type[Marker]) -> Type[Marker]:
        if not issubclass(marker_class, Marker):
            raise RuntimeError("Can only register marker classes as configurable.")
        # to simplify things:
        specified_tags = {tag}
        if negated_tag is not None:
            specified_tags.add(negated_tag)
        # tags should be unique
        for tag_ in specified_tags:
            if tag_ in TAGS:
                raise RuntimeError(
                    "Expected the tags of all configurable markers to be "
                    "identifyable by their tag."
                )
            TAGS.add(tag_)
        # (non-negated) tag <-> class
        TAG_TO_MARKER_CLASS[tag] = marker_class
        MARKER_CLASS_TO_TAG[marker_class] = tag
        # (non-negated) tag <-> negated tag
        if negated_tag is not None:
            NEGATED_TAG_TO_TAG[negated_tag] = tag
            TAG_TO_NEGATED_TAG[tag] = negated_tag
        # condition or operator?
        for type, collection in [
            (AtomicMarker, CONDITION_TAGS),
            (CompoundMarker, OPERATOR_TAGS),
        ]:
            if issubclass(marker_class, type):
                for tag_ in specified_tags:
                    collection.add(tag)
        return marker_class

    return inner


def inspect_tag(tag: Text) -> Tuple[Text, bool]:
    """Given a negative tag, returns the positive version.

    Returns:
        the tag itself if it was already positive, otherwise the positive version;
        and a boolean that represents whether the given tag was a negative one
    """
    # either the given `marker_name` is already "positive" (e.g. "slot was set")
    # or there is an entry mapping it to it's positive version:
    positive_version = NEGATED_TAG_TO_TAG.get(tag, tag)
    is_negation = tag != positive_version
    return positive_version, is_negation


# We allow multiple atomic markers to be grouped under the same tag e.g.
# 'slot_set: ["slot_a", "slot_b"]' (see `AtomicMarkers` / `CompoundMarkers`),
# which is why this config maps to a list:
ConditionConfigList = Dict[Text, List[Text]]
# Compound markers can be nested:
OperatorConfig = Dict[Text, List[Union["OperatorConfig", ConditionConfigList]]]
# In case no compound operator is defined, "and" is used by default. Hence,
# a marker config can also just consist of an atomic marker config list.
MarkerConfig = Union[ConditionConfigList, OperatorConfig]


class InvalidMarkerConfig(RasaException):
    """Exception that can be raised when the config for a marker is not valid."""


MetaData = TypedDict(
    "MetaData", {"preceeding_user_turns": List[int], "timestamp": List[int]}
)

T = TypeVar("T")


class Marker(ABC):
    """A marker is a way of describing points in conversations you're interested in.

    Here, markers are stateful objects because they track the events of a conversation.
    At each point in the conversation, one can observe whether a marker applies or
    does not apply to the conversation so far.
    """

    def __init__(self, name: Optional[Text] = None, negated: bool = False):
        """Instantiates a marker.

        Args:
            name: a custom name that can be used to replace the default string
                conversion of this marker
            negated: whether this marker should be negated (i.e. a negated marker
                applies if and only if the non-negated marker does not apply)
        """
        if name == ANY_MARKER:
            raise InvalidMarkerConfig(
                f"You must not use the special marker name {ANY_MARKER}. This is "
                f"to avoid confusion when you generate a marker from a list of "
                f"marker configurations, which will lead to all markers to be "
                f"combined under one common `ORMarker` named {ANY_MARKER}."
            )
        self.name = name
        self.history: List[bool] = []
        self.negated: bool = negated

    def __str__(self) -> Text:
        return self.name or repr(self)

    def __repr__(self) -> Text:
        tag = MARKER_CLASS_TO_TAG.get(self.__class__, None)
        if tag is None:
            raise RuntimeError(f"This class {self.__class__} is not configurable.")
        if self.negated:
            tag = TAG_TO_NEGATED_TAG.get(tag)
        return self._to_str_with(tag)

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

    def evaluate_events(
        self, events: List[Event], recursive: bool = False
    ) -> Dict[Text, MetaData]:
        """Resets the marker, tracks all events, and collects some information.

        The collected information includes:
        - the timestamp of each event where the marker applied and
        - the number of user turns that preceded that timestamp

        If this marker is the special `ANY_MARKER` (identified by it's name), then
        results will be collected for all (immediate) sub-markers.

        If `recursive` is set to `True`, then all included markers are evaluated.

        Args:
            events: a list of events describing a conversation
            recursive: set this to `True` to collect evaluations for all markers that
               this marker consists of
        """
        # determine which marker to extract results from
        markers_to_be_evaluated: List[Marker] = []
        if not recursive:
            if isinstance(self, CompoundMarker) and self.name == ANY_MARKER:
                markers_to_be_evaluated = self.sub_markers
            else:
                markers_to_be_evaluated = [self]
        else:
            markers_to_be_evaluated = [marker for marker in self]

        # track all events and collect meta data per timestep
        meta_data = self._track_all_and_collect_meta_data(events=events)

        # for each marker, keep meta data only for those timesteps where it applies
        return self._filter_all(markers=markers_to_be_evaluated, meta_data=meta_data)

    def _track_all_and_collect_meta_data(self, events: List[Event]) -> MetaData:
        """Resets the marker, tracks all events, and collects metadata.

        Args:
            events: all events that should be tracked
        Returns:
            metadata for each tracked event
        """
        self.reset()
        timestamps: List[int] = []
        preceeding_user_turns: List[int] = [0]
        for event in events:
            is_user_turn = isinstance(event, UserUttered)
            preceeding_user_turns.append(preceeding_user_turns[-1] + int(is_user_turn))
            timestamps.append(event.timestamp)
            self.track(event=event)
        preceeding_user_turns = preceeding_user_turns[:-1]  # drop last
        return {
            "preceeding_user_turns": preceeding_user_turns,
            "timestamp": timestamps,
        }

    @classmethod
    def _filter_all(
        cls, markers: List[Marker], meta_data: MetaData,
    ) -> Dict[Text, MetaData]:
        """Filters the given meta data according to where the respective marker applies.

        Args:
            markers: the markers that we want to use for filtering the meta data
            meta_data: some meta data with one item per event that was tracked by
                each of the given markers
        Returns:
            a dictionary mapping the string respresentation of the respective marker
            to the filtered meta data
        """
        results: Dict[Text, MetaData] = dict()
        for marker in markers:
            marker_results = {
                key: marker._filter(meta_data_per_event)
                for key, meta_data_per_event in meta_data.items()
            }
            results[str(marker)] = marker_results
        return results

    def _filter(self, items: List[T]) -> List[T]:
        """Returns the items for the points in time where the marker applies.

        Args:
            marker: a marker that has tracked some events
            items: a list of items for each tracked event
        Returns:
            a dictionary mapping all applied filters to a list containing meta data
            for the i-th event if the respective marker applies after the i-th tracked
            event
        """
        if len(self.history) != len(items):
            raise RuntimeError(
                f"Expected the marker to have tracked {len(items)} many events "
                f"but only found {len(self.history)}"
            )
        return [
            meta_data_for_event
            for applies, meta_data_for_event in zip(self.history, items)
            if applies
        ]

    @classmethod
    def from_config_file(cls, path: Union[Text, Path]) -> Marker:
        """Loads markers from a single config file and combines them into one Marker.

        A config file for `Markers` is a dictionary mapping custom marker names
        to configurations of a `CompoundMarker` or `AtomicMarker`.

        For more details, see `from_config_dict`.

        Args:
            path: the path to a config file
        Returns:
            all configured markers, combined into one marker object
        """
        path = os.path.abspath(path)
        config = rasa.shared.io.read_yaml_file(path)
        return cls.from_config(config)

    @classmethod
    def from_all_config_files_in_directory_tree(cls, path: Union[Text, Path]) -> Marker:
        """Loads and appends multiple configs from a directory tree.

        For more details, see `from_config_dict`.

        Args:
            path: the root of the directory tree that contains marker config files
        Returns:
            all configured markers, combined into one marker object
        """
        combined_configs = {}
        path = os.path.abspath(path)
        for root, _, files in os.walk(path, followlinks=True):
            for file in files:
                full_path = os.path.join(root, file)
                if is_likely_yaml_file(full_path):
                    config = rasa.shared.io.read_yaml_file(full_path)
                    cls.validate_config(config)
                    if set(config.keys()).intersection(combined_configs.keys()):
                        raise InvalidMarkerConfig(
                            f"The names of markers defined in {full_path} "
                            f"({sorted(config.keys())}) "
                            f"overlap with the names of markers loaded so far "
                            f"({sorted(combined_configs.keys())})."
                        )
                    combined_configs.extend(config)
                    logging.info(f"Added markers from {full_path}")
        if not config:
            raise InvalidMarkerConfig(
                f"Could not load any markers from the directory tree rooted at {path}."
            )
        return cls.from_config(config)

    @classmethod
    def from_config_dict(cls, config: Dict[Text, MarkerConfig]) -> Marker:
        """Creates markers from a dictionary of marker configurations.

        If there is more than one custom marker defined this way, then the returned
        marker will be an `or` combination of all custom markers named `ANY_MARKER`.
        When evaluating this marker, the results for the special `ANY_MARKER` will
        be ignored and only the results for the custom markers will be returned.

        Args:
            config: mapping custom marker names to marker configurations
        Returns:
            all configured markers, combined into one marker
        """
        from rasa.core.evaluation.marker import OrMarker

        markers = [
            cls.from_config(marker_config, name=marker_name)
            for marker_name, marker_config in config.items()
        ]
        if len(markers) > 1:
            marker = OrMarker(markers=markers)
            marker.name = ANY_MARKER  # cannot be set via name parameter
        else:
            marker = markers[0]
        return marker

    @classmethod
    def from_config(cls, config: MarkerConfig, name: Optional[Text] = None) -> Marker:
        """Creates a marker from the given config.

        Args:
            config: the configuration of a single or multiple markers
            name: a custom name that will be used for the top-level marker (if and
                only if there is only one top-level marker)

        Returns:
            the configured marker
        """
        from rasa.core.evaluation.marker import AndMarker

        # A marker config can be either an atomic marker config list or a
        # compound marker configuration - if it is not a list, we
        marker_configs = config if isinstance(config, list) else [config]

        collected_sub_markers = []
        for marker_config in marker_configs:

            # FIXME: check that this is basically a tuple
            marker_name = next(iter(marker_config))
            sub_marker_config = marker_config[marker_name]

            if marker_name in OPERATOR_TAGS:
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
                    sub_marker_configs=sub_marker_config,
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
    ):
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

    def reset(self) -> None:
        """Evaluate this marker given the next event.

        Args:
            event: the next event of the conversation
        """
        for marker in self.sub_markers:
            marker.reset()
        super().reset()

    @classmethod
    def from_config(
        cls,
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
        if operator not in OPERATOR_TAGS:
            raise InvalidConfigException(f"Unknown operator '{operator}'.")

        positive_version, is_negation = inspect_tag(operator)

        collected_sub_markers: List[Marker] = []
        for sub_marker_name_and_config in sub_marker_configs:
            sub_marker_name = next(iter(sub_marker_name_and_config))
            sub_marker_config = sub_marker_name_and_config[sub_marker_name]

            if sub_marker_name in CONDITION_TAGS:
                # We allow several AtomicMarkers to be collected under the same tag.
                # If this is done at the top-level (see `Marker.from_config`), then
                # we would combine them under a new `AndMarker`.
                # Here, we do *not* create a new `AndMarker` but add the single
                # atomic markers to this compound marker.
                next_sub_markers = AtomicMarker.from_config(
                    marker_name=sub_marker_name, sub_marker_configs=sub_marker_config
                )
            else:
                next_sub_markers = [
                    CompoundMarker.from_config(
                        operator=sub_marker_name, sub_marker_configs=sub_marker_config
                    )
                ]
            collected_sub_markers.extend(next_sub_markers)

        operator_class: Type[CompoundMarker] = TAG_TO_MARKER_CLASS[positive_version]
        marker = operator_class(markers=collected_sub_markers, negated=is_negation)
        marker.name = name
        return marker


class AtomicMarker(Marker, ABC):
    """A marker that does not contain any sub-markers."""

    def __init__(self, text: Text, negated: bool = False, name: Optional[Text] = None):
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

    @classmethod
    def from_config(
        cls,
        marker_name: Text,
        sub_marker_configs: List[Text],
        name: Optional[Text] = None,
    ) -> List[Marker]:
        """Creates an atomic marker from the given config.

        # TODO: describe expected config

        Args:
            marker_name: string identifying an atomic marker type
            sub_marker_configs: a list of text parameter passed to the atomic markers
            name: a custom name for this marker
        Returns:
            the configured `AtomicMarker`s
        """
        if marker_name not in CONDITION_TAGS:
            raise InvalidConfigException(f"Unknown condition '{marker_name}'.")
        positive_version, is_negation = inspect_tag(marker_name)
        marker_class = TAG_TO_MARKER_CLASS[positive_version]
        markers = []
        for marker_text in sub_marker_configs:
            marker = marker_class(marker_text, negated=is_negation)
            marker.name = name
            markers.append(marker)
        return markers
