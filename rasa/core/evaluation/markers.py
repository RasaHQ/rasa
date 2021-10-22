from __future__ import annotations
import os
import sys
from abc import ABC, abstractmethod
from enum import Enum
from typing import (
    Dict,
    Iterator,
    Optional,
    Text,
    List,
    Type,
    Union,
)

if sys.version_info >= (3, 8):
    from typing import TypedDict
else:
    from typing_extensions import TypedDict
from pathlib import Path

import rasa.shared.core.constants
import rasa.shared.nlu.constants
import rasa.shared.utils.validation
import rasa.shared.utils.io
import rasa.shared.utils.common
from rasa.shared.data import is_likely_yaml_file
from rasa.shared.nlu.constants import INTENT_NAME_KEY
from rasa.shared.exceptions import InvalidConfigException, RasaException
from rasa.shared.core.events import ActionExecuted, SlotSet, UserUttered, Event

import logging

logger = logging.getLogger(__name__)

ANY_MARKER = "<any_marker>"
NOT = "_not_"


class AtomicMarkerOptions(str, Enum):
    """Names of atomic markers.

    As a shortcut, you can use negated versions of these atomic marker options (instead
    of creating a compound marker that negates a single atomic marker).
    Just replace the first "_" by "_not_".
    """

    # Reminder: The atomic marker names must not include the substring "_not_".
    ACTION_EXECUTED = "action_executed"
    INTENT_DETECTED = "intent_detected"
    SLOT_SET = "slot_set"


def _is_negated_atomic_marker_name(
    marker_name: Union[Text, AtomicMarkerOptions]
) -> bool:
    """Checks wether the given name is a negated atomic marker name."""
    return NOT in str(marker_name) and any(
        marker_name == _negate_atomic_marker_name(marker_name)
        for marker_name in AtomicMarkerOptions
    )


def _negate_atomic_marker_name(atomic_marker: Union[Text, AtomicMarkerOptions]) -> Text:
    """Returns the negated version of an atomic marker option."""
    if _is_negated_atomic_marker_name(atomic_marker):
        return str(atomic_marker).replace(NOT, "_", 1)
    else:
        return str(atomic_marker).replace("_", NOT, 1)


ATOMIC_MARKERS_AND_NEGATIONS = set().union(
    *[
        {atomic_marker, _negate_atomic_marker_name(atomic_marker)}
        for atomic_marker in AtomicMarkerOptions
    ]
)


class CompoundOptions(str, Enum):
    """Constants that can be used in configs to describe combinations of markers."""

    AND = "and"
    OR = "or"
    NOT = "not"
    SEQUENCE = "seq"


EvaluationResult = TypedDict(
    "EvaluationResult", {"preceeding_user_turns": List[int], "timestamp": List[int]}
)


class InvalidMarkerConfig(RasaException):
    """Exception that can be raised when the config for a marker is not valid."""


MarkerConfig = Union[
    List["MarkerConfig"], Dict[Text, "MarkerConfig"], Dict[Text, List[Text]]
]


class Marker(ABC):
    """A marker is a way of describing points in conversations you're interested in.

    Here, markers are stateful objects because they track the events of a conversation.
    At each point in the conversation, one can observe whether a marker applies or
    does not apply to the conversation so far.
    """

    def __init__(self, name: Optional[Text] = None):
        """Instantiates a marker.

        Args:
            name: a custom name that can be used to replace the default string
                conversion of this marker
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

    def __str__(self) -> Text:
        return self.name or repr(self)

    def track(self, event: Event) -> None:
        """Updates the marker according to the given event.

        Args:
            event: the next event of the conversation
        """
        ...

    def reset(self) -> None:
        """Clears the history of the marker.

        """
        ...

    @abstractmethod
    def __iter__(self) -> Iterator[Marker]:
        """Returns all markers that are part of this marker.

        Returns:
            an iterator over all markers that are part of this marker.
        """
        ...

    def evaluate_events(self, events: List[Event]) -> Dict[Text, EvaluationResult]:
        """Resets the marker, tracks all events, and collects results.

        Args:
            events: a list of events describing a conversation
        """
        self.reset()
        timestamps: List[int] = []
        preceeding_user_turns: List[int] = [0]
        for _, event in enumerate(events):
            is_user_turn = isinstance(event, UserUttered)
            preceeding_user_turns.append(preceeding_user_turns[-1] + is_user_turn)
            timestamps.append(event.timestamp)
            self.track(event=event)
        preceeding_user_turns = preceeding_user_turns[:-1]  # drop last

        results: Dict[Text, EvaluationResult] = dict()
        for sub_marker in self:
            if len(sub_marker.history) != len(timestamps):
                raise RuntimeError("We forgot to update some marker.")

            marker_user_turns = [
                num_user_turns
                for applies, num_user_turns in zip(
                    sub_marker.history, preceeding_user_turns
                )
                if applies
            ]
            if marker_user_turns:
                marker_timestamps: List[int] = [
                    timestamp
                    for applies, timestamp in zip(sub_marker.history, timestamps)
                    if applies
                ]
                results[str(sub_marker)] = {
                    "preceeding_user_turns": marker_user_turns,
                    "timestamp": marker_timestamps,
                }
        # TODO: filter results?
        return results

    @classmethod
    def from_config_file(cls, path: Union[Text, Path]) -> Marker:
        """Loads markers from a single config file and combines them into one Marker.

        Args:
            path:
        """
        path = os.path.abspath(path)
        config = rasa.shared.io.read_yaml_file(path)
        return cls.from_config(config)

    @classmethod
    def from_all_config_files_in_directory_tree(cls, path: Union[Text, Path]) -> Marker:
        """Loads and appends multiple configs from a directory tree.

        Args:
            path:
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
    def validate_config(cls, config: Dict) -> None:
        """Validate the schema of the config.

        Args:
            config: a configuration used to instantiate markers via `Marker.from_config`
        """
        # TODO
        # if not self._is_marker_config() .... / schema
        ...

    @classmethod
    def from_config_dict(cls, config: Dict[Text, MarkerConfig]) -> Marker:
        """Creates markers from a dictionary of marker configurations.

        If there is more than one marker, then all markers will be combined into one
        top level marker ('or') which will be evaluated as `ANY_MARKER`.

        Args:
            config: mapping custom marker names to marker configurations
        Returns:
            all configured markers, combined into one marker
        """
        cls.validate_config(config)

        markers = [
            cls.from_config(marker_config, name=marker_name)
            for marker_name, marker_config in config.items()
        ]
        if len(markers) > 1:
            marker = OrMarker(markers=markers)
            marker.name = ANY_MARKER
        else:
            marker = markers[0]
        return marker

    @classmethod
    def from_config(
        cls, config: Dict[Text, List[MarkerConfig]], name: Optional[Text] = None
    ) -> Marker:
        """Creates a marker from the given config.

        Args:
            config: the configuration of a single or multiple markers
            name: a custom name that will be used for the top-level marker (if and
                only if there is only one top-level marker)
        Returns:
            the configured marker
        """
        assert len(config) == 1  # FIXME: should just be a tuple
        key = next(iter(config.keys()))
        sub_marker_configs = config[key]
        if not isinstance(sub_marker_configs, list):
            raise InvalidMarkerConfig(
                f"Expected a list as sub-marker configuration of {key} but "
                f"received {sub_marker_configs}"
            )
        if any(operator in config.keys() for operator in CompoundOptions):
            return CompoundMarker.from_config(
                operator=key, sub_marker_configs=sub_marker_configs, name=name
            )
        else:
            return AtomicMarker.from_config(
                marker_name=key, sub_marker_configs=sub_marker_configs, name=name
            )


class CompoundMarker(Marker, ABC):
    """Combines several markers into one."""

    def __init__(self, markers: List[Marker], name: Optional[Text] = None):
        """Instantiates a marker.

        Args:
            markers: the list of markers to combine
            name: a custom name that can be used to replace the default string
                conversion of this marker
        """
        super().__init__(name=name)
        self.sub_markers: List[Marker] = markers

    def track(self, event: Event) -> None:
        """Updates the marker according to the given event.

        All sub-markers will be updated before the compound marker itself is updated.

        Args:
            event: the next event of the conversation
        """
        for marker in self.sub_markers:
            marker.track(event)
        marker_applies = self._track(event)
        self.history.append(marker_applies)

    def __iter__(self) -> Iterator[Marker]:
        """Returns all Markers that are part of this Marker (including itself).

        Returns:
            a list of all markers that this marker consists of, which should be
            updated and evaluated
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
        operator: CompoundOptions,
        sub_marker_configs: List[MarkerConfig],
        name: Optional[Text] = None,
    ) -> Marker:
        """Creates a compound marker from the given config.

        Args:
            operator: a text identifying a compound marker type
            sub_marker_config: a list of configs defining sub-markers
            name: an optional custom name to be attached to the resulting marker
        Returns:
           a compound marker if there are markers to combine - and just an event
           marker if there is only a single marker
        """

        operator_to_compound: Dict[Text, Type[CompoundMarker]] = {
            CompoundOptions.AND: AndMarker,
            CompoundOptions.OR: OrMarker,
            CompoundOptions.NOT: NotAnyMarker,
            CompoundOptions.SEQUENCE: SequenceMarker,
        }

        sub_markers = []
        for sub_marker_config in sub_marker_configs:
            sub_marker = Marker.from_config(sub_marker_config)
            sub_markers.append(sub_marker)

        if len(sub_markers) == 1:
            return sub_markers[0]

        compound_marker_type = operator_to_compound.get(operator, None)
        if not compound_marker_type:
            raise RuntimeError("Unknown combination of markers: ")

        return compound_marker_type(markers=sub_markers, name=name)


class AndMarker(CompoundMarker):
    """Checks that all sub-markers apply."""

    def __repr__(self) -> Text:
        return "({})".format(" and ".join(str(marker) for marker in self.sub_markers))

    def _track(self, event: Event) -> bool:
        return all(marker.history[-1] for marker in self.sub_markers)


class OrMarker(CompoundMarker):
    """Checks that one sub-markers is applies."""

    def __repr__(self) -> Text:
        return "({})".format(" or ".join(str(marker) for marker in self.sub_markers))

    def _track(self, event: Event) -> bool:
        return any(marker.history[-1] for marker in self.sub_markers)


class NotAnyMarker(CompoundMarker):
    """Checks that none of the sub-markers applies."""

    def __repr__(self) -> Text:
        if len(self.sub_markers) > 1:
            sub_markers_str = " or ".join(str(marker) for marker in self.sub_markers)
            return f"{CompoundOptions.NOT}({sub_markers_str})"
        else:
            # it must be an AtomicMarker then and by design there is a way
            # to map that AtomicMarker name to a negated one
            return _negate_atomic_marker_name(repr(self.sub_markers[0]))

    def _track(self, event: Event) -> bool:
        return not any(marker.history[-1] for marker in self.sub_markers)


class SequenceMarker(CompoundMarker):
    """Checks the sub-markers applied in the given order."""

    def __repr__(self) -> Text:
        sub_markers_str = " > ".join(str(marker) for marker in self.sub_markers)
        return f"{CompoundOptions.SEQUENCE}({sub_markers_str})"

    def _track(self, event: Event) -> bool:
        # Note: the sub-markers have been updated before this tracker
        if len(self.history) < len(self.sub_markers) - 1:
            return False
        return all(
            marker.history[-idx - 1]
            for idx, marker in enumerate(reversed(self.sub_markers))
        )


class AtomicMarker(Marker, ABC):
    """A marker that does not contain any sub-markers."""

    def __init__(self, text: Text, name: Optional[Text] = None):
        super().__init__(name=name)
        self.text = text

    def track(self, event: Event) -> None:
        """Updates the marker according to the given event.

        Args:
            event: the next event of the conversation
        """
        marker_applies = self._track(event)
        self.history.append(marker_applies)

    def __iter__(self) -> Iterator[Marker]:
        yield self

    @classmethod
    def from_config(
        cls,
        marker_name: Text,
        sub_marker_configs: List[Text],
        name: Optional[Text] = None,
    ) -> Marker:
        """Creates an atomic marker from the given config.

        Args:
            marker_name: string identifying an atomic marker type
            sub_marker_configs: the texts for which atomic markers should be created
                (and combined with or)
            name: a custom name that will be used for the top-level marker (if and
                only if there is only one top-level marker)
        Returns:
            an `AtomicMarker` in case the `sub_marker_configs` list contains only
            one text and an `OrMarker` otherwise
        """
        str_to_marker_type: Dict[Text, Type[AtomicMarker]] = {
            AtomicMarkerOptions.ACTION_EXECUTED: ActionExecutedMarker,
            AtomicMarkerOptions.INTENT_DETECTED: IntentDetectedMarker,
            AtomicMarkerOptions.SLOT_SET: SlotSetMarker,
        }

        if marker_name not in ATOMIC_MARKERS_AND_NEGATIONS:
            raise InvalidConfigException(f"Unknown atomic marker name {marker_name}.")
        marker_type_str = marker_name.replace("_not_", "_")
        marker_type = str_to_marker_type.get(marker_type_str, None)

        markers = []
        for marker_text in sub_marker_configs:
            marker = marker_type(marker_text)
            if _is_negated_atomic_marker_name(marker_name):
                marker = NotAnyMarker([marker], name=name)
            markers.append(marker)

        if len(markers) > 1:
            final_marker = AndMarker(markers=markers, name=name)
        else:
            final_marker = markers[0]

        return final_marker


class ActionExecutedMarker(AtomicMarker):
    """Checks whether an action is executed at the current step."""

    def __repr__(self) -> Text:
        return f"({AtomicMarkerOptions.ACTION_EXECUTED}: {self.text})"

    def _track(self, event: Event) -> bool:
        return isinstance(event, ActionExecuted) and event.action_name == self.text


class IntentDetectedMarker(AtomicMarker):
    """Checks whether an intent is expressed at the current step."""

    def __repr__(self) -> Text:
        return f"({AtomicMarkerOptions.INTENT_DETECTED}: {self.text})"

    def _track(self, event: Event) -> bool:
        return (
            isinstance(event, UserUttered)
            and event.intent.get(INTENT_NAME_KEY) == self.text
        )


class SlotSetMarker(AtomicMarker):
    """Checks whether a slot is set at the current step.

    The actual `SlotSet` event might have happened at an earlier step.
    """

    def __repr__(self) -> Text:
        return f"({AtomicMarkerOptions.ACTION_EXECUTED}: {self.text})"

    def _track(self, event: Event) -> bool:
        if isinstance(event, SlotSet) and event.key == self.text:
            # it might be un-set
            return event.value is not None
        else:
            # it is still set
            return bool(len(self.history) and self.history[-1])
