import logging
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Text, Union

from rasa.e2e_test.assertions import Assertion
from rasa.e2e_test.constants import (
    KEY_ASSERTIONS,
    KEY_ASSERTION_ORDER_ENABLED,
    KEY_BOT_INPUT,
    KEY_BOT_UTTERED,
    KEY_FIXTURES,
    KEY_METADATA,
    KEY_STUB_CUSTOM_ACTIONS,
    KEY_SLOT_NOT_SET,
    KEY_SLOT_SET,
    KEY_STEPS,
    KEY_TEST_CASE,
    KEY_TEST_CASES,
    KEY_USER_INPUT,
)
from rasa.e2e_test.stub_custom_action import StubCustomAction
from rasa.shared.core.events import BotUttered, SlotSet, UserUttered
from rasa.shared.exceptions import RasaException

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Fixture:
    """Class for storing an input fixture."""

    name: Text
    slots_set: Dict[Text, Any]

    @staticmethod
    def from_dict(fixture_dict: Dict[Text, Any]) -> "Fixture":
        """Creates a fixture from a dictionary.

        Example:
            >>> Fixture.from_dict({"some_fixture": [{"slot_name": "slot_value"}]})
            Fixture(name="some_fixture", slots_set={"slot_name": "slot_value"})

        Args:
            fixture_dict: Dictionary containing the fixture.
        """
        return Fixture(
            name=next(iter(fixture_dict.keys())),
            slots_set={
                slot_name: slot_value
                for slots_list in fixture_dict.values()
                for slot_dict in slots_list
                for slot_name, slot_value in slot_dict.items()
            },
        )

    def as_dict(self) -> Dict[Text, Any]:
        """Returns the fixture as a dictionary."""
        return {
            self.name: [
                {slot_name: slot_value}
                for slot_name, slot_value in self.slots_set.items()
            ]
        }


@dataclass(frozen=True)
class TestStep:
    """Class for storing the input test step.

    Instances are frozen to make sure the underlying dictionary stays in sync
    with the text and actor attributes.
    """

    actor: Text
    text: Optional[Text] = None
    template: Optional[Text] = None
    line: Optional[int] = None
    slot_was_set: bool = False
    slot_was_not_set: bool = False
    _slot_instance: Optional[Union[Text, Dict[Text, Any]]] = None
    _underlying: Optional[Dict[Text, Any]] = None
    metadata_name: Optional[Text] = None
    assertions: Optional[List[Assertion]] = None
    assertion_order_enabled: bool = False

    @staticmethod
    def from_dict(test_step_dict: Dict[Text, Any]) -> "TestStep":
        """Creates a test step from a dictionary.

        Example:
            >>> TestStep.from_dict({"user": "hello"})
            TestStep(text="hello", actor="user")
            >>> TestStep.from_dict({"user": "hello", "metadata": "some_metadata"})
            TestStep(text="hello", actor="user", metadata_name="some_metadata")
            >>> TestStep.from_dict({"bot": "hello world"})
            TestStep(text="hello world", actor="bot")

        Args:
            test_step_dict: Dictionary containing the test step.
        """
        TestStep._validate_input_dict(test_step_dict)
        slot_instance = test_step_dict.get(KEY_SLOT_SET)
        if test_step_dict.get(KEY_SLOT_NOT_SET):
            slot_instance = test_step_dict.get(KEY_SLOT_NOT_SET)

        assertions = (
            [
                Assertion.create_typed_assertion(data)
                for data in test_step_dict.get(KEY_ASSERTIONS, [])
            ]
            if KEY_ASSERTIONS in test_step_dict
            else None
        )

        return TestStep(
            text=test_step_dict.get(
                KEY_USER_INPUT, test_step_dict.get(KEY_BOT_INPUT, "")
            ).strip()
            or None,
            template=test_step_dict.get(KEY_BOT_UTTERED),
            actor=KEY_USER_INPUT if KEY_USER_INPUT in test_step_dict else KEY_BOT_INPUT,
            line=test_step_dict.lc.line + 1 if hasattr(test_step_dict, "lc") else None,
            slot_was_set=bool(test_step_dict.get(KEY_SLOT_SET)),
            slot_was_not_set=bool(test_step_dict.get(KEY_SLOT_NOT_SET)),
            _slot_instance=slot_instance,
            _underlying=test_step_dict,
            metadata_name=test_step_dict.get(KEY_METADATA, ""),
            assertions=assertions,
            assertion_order_enabled=test_step_dict.get(
                KEY_ASSERTION_ORDER_ENABLED, False
            ),
        )

    @staticmethod
    def _validate_input_dict(test_step_dict: Dict[Text, Any]) -> None:
        if (
            KEY_USER_INPUT not in test_step_dict
            and KEY_BOT_INPUT not in test_step_dict
            and KEY_BOT_UTTERED not in test_step_dict
            and KEY_SLOT_SET not in test_step_dict
            and KEY_SLOT_NOT_SET not in test_step_dict
        ):
            raise RasaException(
                f"Test step is missing either the {KEY_USER_INPUT}, {KEY_BOT_INPUT}, "
                f"{KEY_SLOT_NOT_SET}, {KEY_SLOT_SET} "
                f"or {KEY_BOT_UTTERED} key: {test_step_dict}"
            )

        if (
            test_step_dict.get(KEY_SLOT_SET) is not None
            and test_step_dict.get(KEY_SLOT_NOT_SET) is not None
        ):
            raise RasaException(
                f"Test step has both {KEY_SLOT_SET} and {KEY_SLOT_NOT_SET} keys: "
                f"{test_step_dict}. You must only use one of the keys in a test step."
            )

        if KEY_USER_INPUT not in test_step_dict and KEY_ASSERTIONS in test_step_dict:
            raise RasaException(
                f"Test step with assertions must only be used with the "
                f"'{KEY_USER_INPUT}' key: {test_step_dict}"
            )

        if (
            KEY_USER_INPUT not in test_step_dict
            and KEY_ASSERTION_ORDER_ENABLED in test_step_dict
        ):
            raise RasaException(
                f"Test step with '{KEY_ASSERTION_ORDER_ENABLED}' key must "
                f"only be used with the '{KEY_USER_INPUT}' key: "
                f"{test_step_dict}"
            )

        if (
            KEY_ASSERTION_ORDER_ENABLED in test_step_dict
            and KEY_ASSERTIONS not in test_step_dict
        ):
            raise RasaException(
                f"You must specify the '{KEY_ASSERTIONS}' key in the user test step "
                f"where you are using '{KEY_ASSERTION_ORDER_ENABLED}' key: "
                f"{test_step_dict}"
            )

    def as_dict(self) -> Dict[Text, Any]:
        """Returns the underlying dictionary of the test step."""
        return self._underlying or {}

    def as_dict_yaml_format(self) -> Dict[Text, Any]:
        """Returns the test step as a dictionary in YAML format."""
        if not self._underlying:
            return {}

        result = self._underlying.copy()

        def _handle_slots(key: str) -> None:
            """Slots should be a list of strings or dicts."""
            if (
                self._underlying
                and key in self._underlying
                and isinstance(self._underlying[key], OrderedDict)
            ):
                result[key] = [
                    {key: value} for key, value in self._underlying[key].items()
                ]
            elif (
                self._underlying
                and key in self._underlying
                and isinstance(self._underlying[key], str)
            ):
                result[key] = [self._underlying[key]]

        _handle_slots(KEY_SLOT_SET)
        _handle_slots(KEY_SLOT_NOT_SET)

        return result

    def matches_event(self, other: Union[BotUttered, SlotSet, None]) -> bool:
        """Compares the test step with BotUttered or SlotSet event.

        Args:
            other: BotUttered or SlotSet event to compare with.

        Returns:
            `True` if the event and step match, `False` otherwise.
        """
        if other is None or self.actor != "bot":
            return False

        if isinstance(other, BotUttered):
            return self._do_utterances_match(other)
        elif isinstance(other, SlotSet):
            return self._do_slots_match(other)

        return False

    def _do_utterances_match(self, other: BotUttered) -> bool:
        if self.text:
            return other.text == self.text
        elif self.template:
            # we do set this utter_action metadata in the `ActionBotResponse`
            # action, so we can check if the template is the same
            return other.metadata.get("utter_action") == self.template

        return False

    def _do_slots_match(self, other: SlotSet) -> bool:
        if self._slot_instance:
            slot_name, slot_value = self.get_slot_name(), self.get_slot_value()
            if isinstance(self._slot_instance, dict):
                return other.key == slot_name and other.value == slot_value
            return other.key == slot_name

        return False

    def get_slot_name(self) -> Optional[Text]:
        """Returns the slot name if the test step is a slot action."""
        if isinstance(self._slot_instance, str):
            return self._slot_instance
        if isinstance(self._slot_instance, dict):
            return next(iter(self._slot_instance.keys()))
        return None

    def get_slot_value(self) -> Optional[Any]:
        """Returns the slot value if the test step is a slot action."""
        if isinstance(self._slot_instance, dict):
            return next(iter(self._slot_instance.values()))
        return None

    def is_slot_instance_dict(self) -> bool:
        """Asserts if the slot instance is a dictionary."""
        return isinstance(self._slot_instance, dict)


@dataclass
class ActualStepOutput:
    """Class for storing the events that are generated as a response from the bot."""

    actor: Text
    text: Optional[Text]
    events: List[Union[BotUttered, UserUttered, SlotSet]]
    bot_uttered_events: List[BotUttered]
    user_uttered_events: List[UserUttered]
    slot_set_events: List[SlotSet]

    @staticmethod
    def from_test_step(
        test_step: TestStep,
        events: List[Union[BotUttered, UserUttered, SlotSet]],
    ) -> "ActualStepOutput":
        """Creates a test step response from a test step and events.

        Example:
            >>> ActualStepOutput.from_test_step(TestStep(text="hello", actor="bot"))
            ActualStepOutput(text="hello", actor="bot")

        Args:
            test_step: Test step.
            events: Events generated by the bot.
        """
        bot_uttered_events, user_uttered_events, slot_set_events = [], [], []
        for event in events or []:
            if isinstance(event, BotUttered):
                bot_uttered_events.append(event)
            elif isinstance(event, UserUttered):
                user_uttered_events.append(event)
            elif isinstance(event, SlotSet):
                slot_set_events.append(event)

        return ActualStepOutput(
            text=test_step.text,
            actor=test_step.actor,
            events=events,
            bot_uttered_events=bot_uttered_events,
            user_uttered_events=user_uttered_events,
            slot_set_events=slot_set_events,
        )

    def remove_bot_uttered_event(self, bot_utter: BotUttered) -> None:
        """Removes bot_uttered_event from the ActualStepOutput."""
        if bot_utter in self.bot_uttered_events:
            self.bot_uttered_events.remove(bot_utter)
            self.events.remove(bot_utter)

    def remove_user_uttered_event(self, user_utter: UserUttered) -> None:
        """Removes user_uttered_event from the ActualStepOutput."""
        if user_utter in self.user_uttered_events:
            self.user_uttered_events.remove(user_utter)
            self.events.remove(user_utter)

    def remove_slot_set_event(self, slot_set: SlotSet) -> None:
        """Removes slot_set_event from the ActualStepOutput."""
        if slot_set in self.slot_set_events:
            self.slot_set_events.remove(slot_set)
            self.events.remove(slot_set)

    def get_user_uttered_event(self) -> Optional[UserUttered]:
        """Returns the user_uttered_event from the ActualStepOutput."""
        if self.user_uttered_events:
            # we assume that there is only one user_uttered_event
            # in the ActualStepOutput since the response is
            # only from one TestStep with one user utterance
            try:
                return self.user_uttered_events[0]
            except IndexError:
                logger.debug(
                    f"Could not find `UserUttered` event in the ActualStepOutput: "
                    f"{self}"
                )
                return None
        return None


@dataclass
class TestCase:
    """Class for storing the input test case."""

    name: Text
    steps: List[TestStep]
    file: Optional[Text] = None
    line: Optional[int] = None
    fixture_names: Optional[List[Text]] = None
    metadata_name: Optional[Text] = None

    @staticmethod
    def from_dict(
        input_test_case: Dict[Text, Any], file: Optional[Text] = None
    ) -> "TestCase":
        """Creates a test case from a dictionary.

        Example:
            >>> TestCase.from_dict({"test_case": "test", "steps": [{"user": "hello"}]})
            TestCase(name="test", steps=[TestStep(text="hello", actor="user")])

        Args:
            input_test_case: Dictionary containing the test case.
            file: File name of the test case.

        Returns:
            Test case object.
        """
        steps = []

        for step in input_test_case.get(KEY_STEPS, []):
            if KEY_SLOT_SET in step:
                for slot in step[KEY_SLOT_SET]:
                    step_copy = step.copy()
                    step_copy[KEY_SLOT_SET] = slot
                    steps.append(TestStep.from_dict(step_copy))
            elif KEY_SLOT_NOT_SET in step:
                for slot in step[KEY_SLOT_NOT_SET]:
                    step_copy = step.copy()
                    step_copy[KEY_SLOT_NOT_SET] = slot
                    steps.append(TestStep.from_dict(step_copy))
            else:
                steps.append(TestStep.from_dict(step))

        return TestCase(
            name=input_test_case.get(KEY_TEST_CASE, "default"),
            steps=steps,
            file=file,
            line=(
                input_test_case.lc.line + 1 if hasattr(input_test_case, "lc") else None
            ),
            fixture_names=input_test_case.get(KEY_FIXTURES),
            metadata_name=input_test_case.get(KEY_METADATA),
        )

    def as_dict(self) -> Dict[Text, Any]:
        """Returns the test case as a dictionary."""
        result = {
            KEY_TEST_CASE: self.name,
            KEY_STEPS: [step.as_dict_yaml_format() for step in self.steps],
        }
        if self.fixture_names:
            result[KEY_FIXTURES] = self.fixture_names
        if self.metadata_name:
            result[KEY_METADATA] = self.metadata_name
        return result

    def file_with_line(self) -> Text:
        """Returns the file name and line number of the test case."""
        if not self.file:
            return ""

        line = str(self.line) if self.line is not None else ""
        return f"{self.file}:{line}"

    def uses_assertions(self) -> bool:
        """Checks if the test case uses assertions."""
        try:
            next(step for step in self.steps if step.assertions is not None)
        except StopIteration:
            return False

        return True


@dataclass(frozen=True)
class Metadata:
    """Class for storing an input metadata."""

    name: Text
    metadata: Dict[Text, Any]

    @staticmethod
    def from_dict(metadata_dict: Dict[Text, Any]) -> "Metadata":
        """Creates a metadata from a dictionary.

        Example:
            >>> Metadata.from_dict({"some_metadata": {"room": "test_room"}})
            Metadata(name="some_metadata", metadata={"room": "test_room"})

        Args:
            metadata_dict: Dictionary containing the metadata.
        """
        return Metadata(
            name=next(iter(metadata_dict.keys())),
            metadata={
                metadata_name: metadata_value
                for metadata in metadata_dict.values()
                for metadata_name, metadata_value in metadata.items()
            },
        )

    def as_dict(self) -> Dict[Text, Any]:
        """Returns the metadata as a dictionary."""
        return {self.name: self.metadata}


@dataclass(frozen=True)
class TestSuite:
    """Class for representing all top level test suite keys."""

    test_cases: List[TestCase]
    fixtures: List[Fixture]
    metadata: List[Metadata]
    stub_custom_actions: Dict[Text, StubCustomAction]

    def as_dict(self) -> Dict[Text, Any]:
        """Returns the test suite as a dictionary."""
        return {
            KEY_FIXTURES: [fixture.as_dict() for fixture in self.fixtures],
            KEY_METADATA: [metadata.as_dict() for metadata in self.metadata],
            KEY_STUB_CUSTOM_ACTIONS: {
                key: value.as_dict() for key, value in self.stub_custom_actions.items()
            },
            KEY_TEST_CASES: [test_case.as_dict() for test_case in self.test_cases],
        }
