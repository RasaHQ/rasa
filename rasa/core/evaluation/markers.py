from __future__ import annotations
from abc import ABC
from typing import Dict, Optional, Text, List, Type, Union
import os
from pathlib import Path

from ruamel.yaml.parser import ParserError

import rasa.shared.core.constants
from rasa.shared.exceptions import RasaException, YamlSyntaxException
import rasa.shared.nlu.constants
import rasa.shared.utils.validation
import rasa.shared.utils.io
import rasa.shared.utils.common
from rasa.shared.data import is_likely_yaml_file
from rasa.shared.nlu.constants import INTENT_NAME_KEY
from rasa.shared.exceptions import RasaException
from rasa.shared.core.events import ActionExecuted, SlotSet, UserUttered, Event


# from rasa.shared.utils.schemas.markers import MARKERS_SCHEMA


class InvalidMarkersConfig(RasaException):
    """Exception that can be raised when markers config is not valid."""


class Marker(ABC):
    """A marker is a way of describing points in conversations you're interested in.

    Here, markers are stateful objects because they track the events of a conversation.
    At each point in the conversation, one can observe whether a marker applies or
    does not apply to the conversation so far.
    """

    def __init__(self, name: Optional[Text] = None):
        self.name = name
        self.history: List[bool] = []

    def __str__(self):
        return self.name or repr(self)

    def evaluate(self, event: Event) -> None:
        """Evaluate this marker given the next event.

        Args:
            event: the next event of the conversation
        """
        marker_applies = self._evaluate(event)
        self.history.append(marker_applies)

    def _evaluate(self, event: Event) -> bool:
        """Evaluate this marker given the next event.

        Args:
            event: the next event of the conversation
        """
        ...

    def clear(self):
        """Clears the history of the marker.

        """
        self.history = []

    def flatten(self) -> List[Marker]:
        """Returns all Markers that are part of this Marker.

        Returns:
            a list of all markers that this marker consists of, which should be
            updated and evaluated
        """
        ...

    def evaluate_all(
        self, events: List[Event], clear_history: bool = True
    ) -> Dict[Text, Dict[Text, List[float]]]:
        """Tracks all given events and collects the results.

        Args:
            clear_history:

        """
        if clear_history:
            self.clear()
        timestamps = []
        preceeding_user_turns = [0]
        for _, event in enumerate(events):
            is_user_turn = isinstance(event, UserUttered)
            preceeding_user_turns.append(preceeding_user_turns[-1] + is_user_turn)
            timestamps.append(event.timestamp)
            self.evaluate(event=event)
        preceeding_user_turns = preceeding_user_turns[:-1]  # drop last

        results = dict()
        for sub_marker in self.flatten():
            # FIXME: filter out submarkers that we're interested in ... (later?)
            if len(sub_marker.history) != len(timestamps):
                raise RuntimeError("We forgot to update some marker.")

            sub_marker_results = [
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
        """Loads the config from a file or directory."""  # FIXME: this should tell me things get merged
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
                # FIXME: and if not, then I'll never know this won't be merged
        return combined_configs

    @classmethod
    def is_marker_config(cls, config: Dict) -> bool:
        """Merges multiple marker configs."""
        return "markers" in config.keys()

    @classmethod
    def validate_config(cls, config: Dict, filename: Text = "") -> None:
        return True
        # """Validates the markers config according to the schema."""
        # try:
        #     validate(config, MARKERS_SCHEMA)
        # except ValidationError as e:
        #     e.message += (
        #         f". The file {filename} is invalid according to the markers schema."
        #     )
        #     raise e

    @classmethod
    def from_config(
        cls, config: Union[List[Dict], Dict], name: Optional[Text] = None
    ) -> Marker:
        cls.validate_config(config)

        # FIXME: DAG
        # FIXME: schema -> nested
        # FIXME: constants

        if isinstance(config, list):
            markers = [
                cls.from_config(marker_config, name=marker_name)
                for sub_config in config
                for marker_name, marker_config in sub_config.items()
            ]
            if len(markers) > 1:
                return OrMarker(markers=markers, name=name)
            else:
                marker = markers[0]
                marker.name = name
                return marker

        elif isinstance(config, dict):

            assert len(config) == 1

            if any(operator in config.keys() for operator in ["and", "or", "not"]):

                operator = next(iter(config.keys()))

                sub_markers = []
                for sub_marker, sub_marker_config in config[operator].items():
                    for setting in sub_marker_config:  # list of e.g. action names
                        sub_markers.append(cls.from_config({sub_marker: setting}))

                operator_to_compound: Dict[Text, Type[CompoundMarker]] = {
                    "and": AndMarker,
                    "not": NotAnyMarker,
                    "or": OrMarker,
                }
                compound_marker_type = operator_to_compound.get(operator, None)
                if not compound_marker_type:
                    raise RuntimeError("Unknown combination of markers: ")

                return compound_marker_type(markers=sub_markers, name=name)

            else:

                marker_name = next(iter(config.keys()))
                text = config[marker_name]

                str_to_marker_type = {
                    "action_executed": ActionExecutedMarker,
                    "intent_detected": IntentDetectedMarker,
                    "slot_set": SlotSetMarker,
                }

                marker_type = str_to_marker_type.get(
                    marker_name.replace("_not_", "_"), None
                )
                if not marker_type:
                    raise RuntimeError(f"Unknown kind of marker: {marker_name}")

                marker = marker_type(text)
                if "_not_" in marker_name:
                    marker = NotAnyMarker([marker], name=name)
                return marker

        else:
            raise RuntimeError(f"Unknown config format: {config}")


class CompoundMarker(Marker, ABC):
    def __init__(self, markers: List[Marker], name: Optional[Text] = None):
        super().__init__(name=name)
        self.markers: List[Marker] = markers

    def flatten(self) -> List[Marker]:
        return [self] + super().flatten()

    def evaluate(self, event: Event) -> None:
        # FIXME: DAG
        for marker in self.markers:
            marker.evaluate(event)
        super().evaluate(event=event)

    def flatten(self) -> List[Marker]:
        return [self] + [
            sub_marker for marker in self.markers for sub_marker in marker.flatten()
        ]


class AndMarker(CompoundMarker):
    def __repr__(self) -> Text:
        return "({})".format(" and ".join(str(marker) for marker in self.markers))

    def _evaluate(self, event: Event) -> bool:
        return all(marker.history[-1] for marker in self.markers)


class OrMarker(CompoundMarker):
    def __repr__(self) -> Text:
        return "({})".format(" or ".join(str(marker) for marker in self.markers))

    def _evaluate(self, event: Event) -> bool:
        return all(marker.history[-1] for marker in self.markers)


class NotAnyMarker(CompoundMarker):
    def __repr__(self) -> Text:
        return "none of".join(str(marker) for marker in self.markers)

    def _evaluate(self, event: Event) -> bool:
        return not any(marker.history[-1] for marker in self.markers)


class EventMarker(Marker, ABC):
    def __init__(self, text: Text, name: Optional[Text] = None):
        super().__init__(name=name)
        self.text = text

    def flatten(self) -> List[Marker]:
        return [self]


class ActionExecutedMarker(EventMarker):
    def __repr__(self) -> Text:
        return f"(Action {self.text} executed)"

    def _evaluate(self, event: Event) -> bool:
        return isinstance(event, ActionExecuted) and event.action_name == self.text


class IntentDetectedMarker(EventMarker):
    def __repr__(self) -> Text:
        return f"(user expressed intent {self.text})"

    def _evaluate(self, event: Event) -> bool:
        return (
            isinstance(event, UserUttered)
            and event.intent.get(INTENT_NAME_KEY) == self.text
        )


class SlotSetMarker(EventMarker):
    def __repr__(self) -> Text:
        return f"({self.text} set)"

    def _evaluate(self, event: Event) -> bool:
        if isinstance(event, SlotSet) and event.key == self.text:
            # it might be un-set
            return event.value is not None
        else:
            # it is still set
            return len(self.history) and self.history[-1]
