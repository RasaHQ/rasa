import rasa.core
from rasa.shared.constants import LATEST_TRAINING_DATA_FORMAT_VERSION
from rasa.shared.core.constants import ACTION_SESSION_START_NAME
from rasa.shared.core.domain import Domain
from rasa.shared.core.events import (
    SessionStarted,
    SlotSet,
    UserUttered,
    ActionExecuted,
    DefinePrevUserUtteredFeaturization,
)
from rasa.shared.core.trackers import DialogueStateTracker
from rasa.shared.core.training_data.story_reader.yaml_story_reader import (
    YAMLStoryReader,
)
from rasa.shared.core.training_data.story_writer.yaml_story_writer import (
    YAMLStoryWriter,
)
from rasa.shared.core.training_data.structures import Story
from rasa.shared.nlu.constants import INTENT_NAME_KEY

domain = Domain.load("data/test_moodbot/domain.yml")


def test_session_start_is_not_serialised(domain: Domain):
    tracker = DialogueStateTracker("default", domain.slots)
    # the retrieved tracker should be empty
    assert len(tracker.events) == 0

    # add SlotSet event
    tracker.update(SlotSet("slot", "value"))

    # add the two SessionStarted events and a user event
    tracker.update(ActionExecuted(ACTION_SESSION_START_NAME))
    tracker.update(SessionStarted())
    tracker.update(
        UserUttered("say something", intent={INTENT_NAME_KEY: "some_intent"})
    )
    tracker.update(DefinePrevUserUtteredFeaturization(False))

    YAMLStoryWriter().dumps(
        Story.from_events(tracker.events, "some-story01").story_steps
    )

    expected = f"""version: "{LATEST_TRAINING_DATA_FORMAT_VERSION}"
stories:
- story: some-story01
  steps:
  - slot_was_set:
    - slot: value
  - intent: some_intent
"""

    actual = YAMLStoryWriter().dumps(
        Story.from_events(tracker.events, "some-story01").story_steps
    )
    assert actual == expected


def test_as_story_string_or_statement_with_slot_was_set():
    import rasa.shared.utils.io

    stories = """
    stories:
    - story: hello world
      steps:
      - or:
        - slot_was_set:
            - name: joe
        - slot_was_set:
            - name: bob
      - action: some_action
    """

    reader = YAMLStoryReader()
    yaml_content = rasa.shared.utils.io.read_yaml(stories)

    steps = reader.read_from_parsed_yaml(yaml_content)

    assert len(steps) == 3


def test_cap_length():
    assert (
        rasa.shared.core.training_data.structures._cap_length("mystring", 6) == "mys..."
    )


def test_cap_length_without_ellipsis():
    assert (
        rasa.shared.core.training_data.structures._cap_length(
            "mystring", 3, append_ellipsis=False
        )
        == "mys"
    )


def test_cap_length_with_short_string():
    assert rasa.shared.core.training_data.structures._cap_length("my", 3) == "my"
