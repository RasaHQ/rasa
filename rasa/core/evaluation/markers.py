from __future__ import annotations
from abc import ABC, abstractmethod
from enum import Enum
from typing import (
    Dict,
    Iterable,
    Iterator,
    Optional,
    Text,
    List,
    Tuple,
    Type,
    TypedDict,
    Union,
)
import os
from pathlib import Path

from ruamel.yaml.parser import ParserError

import rasa.shared.core.constants
from rasa.shared.core.trackers import DialogueStateTracker
from rasa.shared.exceptions import RasaException, YamlSyntaxException
import rasa.shared.nlu.constants
import rasa.shared.utils.validation
import rasa.shared.utils.io
import rasa.shared.utils.common
from rasa.shared.data import is_likely_yaml_file
from rasa.shared.nlu.constants import INTENT_NAME_KEY
from rasa.shared.exceptions import RasaException
from rasa.shared.core.events import ActionExecuted, SlotSet, UserUttered, Event

import logging

logger = logging.getLogger(__name__)


class AtomicMarkerOptions(str, Enum):
    """Constants that can be used in configs to describe atomic markers."""

    ACTION_EXECUTED = "action_executed"
    ACTION_NOT_EXECUTED = "action_not_executed"
    INTENT_DETECTED = "intent_detected"
    INTENT_NOT_DETECTED = "intent_not_detected"
    SLOT_SET = "slot_set"
    SLOT_NOT_SET = "slot_not_set"


class CompoundOptions(str, Enum):
    """Constants that can be used in configs to describe combinations of markers."""

    AND = "and"
    OR = "or"
    NOT = "not"


EvaluationResult = TypedDict(
    "EvaluationResult", {"preceeding_user_turns": Tuple[int], "timestamp": Tuple[int]}
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
            # FIXME: filter out submarkers that we're interested in ... (later?)
            if len(sub_marker.history) != len(timestamps):
                raise RuntimeError("We forgot to update some marker.")

            sub_marker_results: List[Tuple[int, int]] = [
                (num_user_turns, timestamp)
                for applies, timestamp, num_user_turns in zip(
                    sub_marker.history, timestamps, preceeding_user_turns
                )
                if applies
            ]
            if sub_marker_results:
                marker_user_turns, marker_timestamps = zip(*sub_marker_results)
                # FIXME: map names back
                results[str(sub_marker)] = {
                    "preceeding_user_turns": marker_user_turns,
                    "timestamp": marker_timestamps,
                }
        return results

    @classmethod
    def from_path(cls, path: Union[Text, Path]) -> Marker:
        """Loads the config from a file or directory.

        If loaded from a directory
        """
        path = os.path.abspath(path)
        if os.path.isfile(path):
            config = cls._load_config_from_yaml(path)
        elif os.path.isdir(path):
            config = cls._load_config_from_directory(path)
        else:
            raise InvalidMarkersConfig(
                "Failed to load markers configuration from '{}'. "
                "File not found!".format(os.path.abspath(path))
            )
        return cls.from_config(config)

    @classmethod
    def _load_config_from_yaml(cls, yaml: Text, filename: Text = "") -> Dict:
        """Loads the config from YAML text after validating it."""
        try:
            config = rasa.shared.utils.io.read_yaml(yaml)
            return config

        except ParserError as e:
            e.filename = filename
            raise YamlSyntaxException(filename, e)

    @classmethod
    def _load_config_from_directory(cls, path: Text) -> Dict:
        """Loads and appends multiple configs from a directory tree."""
        combined_configs = cls.empty_config()
        for root, _, files in os.walk(path, followlinks=True):
            for file in files:
                full_path = os.path.join(root, file)
                if is_likely_yaml_file(full_path):
                    config = cls.from_yaml(full_path)
                    if cls.is_marker_config(config):
                        combined_configs.extend(config["markers"])
        return combined_configs

    @classmethod
    def is_marker_config(cls, config: Dict) -> bool:
        """Merges multiple marker configs."""
        return "markers" in config.keys()

    @classmethod
    def validate_config(cls, config: Dict) -> None:
        """Validate the schema of the config.

        Args:
            config: a configuration used to instantiate markers via `Marker.from_config`
        """
        # TODO
        ...

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
        cls.validate_config(config)

        if isinstance(config, list):

            for sub_config in config:
                if any(operator in sub_config for operator in CompoundOptions) or any(
                    event_marker_name in sub_config
                    for event_marker_name in AtomicMarkerOptions
                ):
                    raise RuntimeError(
                        "Expected top level config to contain custom marker names"
                    )

            markers = [
                cls.from_config(marker_config, name=marker_name)
                for sub_config in config
                for marker_name, marker_config in sub_config.items()
            ]
            if len(markers) > 1:
                return OrMarker(markers=markers, name=name)
            else:
                marker = markers[0]
                if name is not None:
                    marker.name = name
                return marker

        elif isinstance(config, dict):

            assert len(config) == 1

            key = next(iter(config.keys()))
            sub_marker_configs = config[key]
            if not isinstance(sub_marker_configs, list):
                raise RuntimeError("This should be a list")

            if any(operator in config.keys() for operator in ["and", "or", "not"]):

                return CompoundMarker.from_config(
                    operator=key, sub_marker_configs=sub_marker_configs, name=name
                )

            else:

                return AtomicMarker.from_config(
                    marker_name=key, sub_marker_configs=sub_marker_configs, name=name
                )

        else:
            raise RuntimeError(f"Unknown config format: {config}")


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
        return "not-any({})".format(
            " or ".join(str(marker) for marker in self.sub_markers)
        )

    def _track(self, event: Event) -> bool:
        return not any(marker.history[-1] for marker in self.sub_markers)


class SequenceMarker(CompoundMarker):
    """Checks the sub-markers applied in the given order."""

    def __repr__(self) -> Text:
        return "sequence".join(str(marker) for marker in self.sub_markers)

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
        marker_name: AtomicMarkerOptions,
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
        str_to_marker_type = {
            AtomicMarkerOptions.ACTION_EXECUTED: ActionExecutedMarker,
            AtomicMarkerOptions.INTENT_DETECTED: IntentDetectedMarker,
            AtomicMarkerOptions.SLOT_SET: SlotSetMarker,
        }

        marker_type_str = marker_name.replace("_not_", "_")
        marker_type = str_to_marker_type.get(marker_type_str, None)
        if not marker_type:
            raise RuntimeError(f"Unknown kind of marker: {marker_name}")

        markers = []
        for marker_text in sub_marker_configs:
            marker = marker_type(marker_text)
            if "_not_" in marker_name:
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
        return f"(Action {self.text} executed)"

    def _track(self, event: Event) -> bool:
        return isinstance(event, ActionExecuted) and event.action_name == self.text


class IntentDetectedMarker(AtomicMarker):
    """Checks whether an intent is expressed at the current step."""

    def __repr__(self) -> Text:
        return f"(user expressed intent {self.text})"

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
        return f"({self.text} set)"

    def _track(self, event: Event) -> bool:
        if isinstance(event, SlotSet) and event.key == self.text:
            # it might be un-set
            return event.value is not None
        else:
            # it is still set
            return bool(len(self.history) and self.history[-1])
