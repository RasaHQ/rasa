import rasa.utils.io
from rasa.core import training
from rasa.core.domain import Domain
from rasa.core.events import (
    UserUttered,
    ActionExecuted,
    ActionExecutionRejected,
    Form,
    FormValidation,
)
from rasa.core.interpreter import RegexInterpreter
from rasa.core.trackers import DialogueStateTracker
from rasa.core.training.story_reader.markdown_story_reader import MarkdownStoryReader
from rasa.core.training.structures import Story


async def test_persist_and_read_test_story_graph(tmpdir, default_domain):
    graph = await training.extract_story_graph(
        "data/test_stories/stories.md", default_domain
    )
    out_path = tmpdir.join("persisted_story.md")
    rasa.utils.io.write_text_file(graph.as_story_string(), out_path.strpath)

    recovered_trackers = await training.load_data(
        out_path.strpath,
        default_domain,
        use_story_concatenation=False,
        tracker_limit=1000,
        remove_duplicates=False,
    )
    existing_trackers = await training.load_data(
        "data/test_stories/stories.md",
        default_domain,
        use_story_concatenation=False,
        tracker_limit=1000,
        remove_duplicates=False,
    )

    existing_stories = {t.export_stories() for t in existing_trackers}
    for t in recovered_trackers:
        story_str = t.export_stories()
        assert story_str in existing_stories
        existing_stories.discard(story_str)


async def test_persist_and_read_test_story(tmpdir, default_domain):
    graph = await training.extract_story_graph(
        "data/test_stories/stories.md", default_domain
    )
    out_path = tmpdir.join("persisted_story.md")
    Story(graph.story_steps).dump_to_file(out_path.strpath)

    recovered_trackers = await training.load_data(
        out_path.strpath,
        default_domain,
        use_story_concatenation=False,
        tracker_limit=1000,
        remove_duplicates=False,
    )
    existing_trackers = await training.load_data(
        "data/test_stories/stories.md",
        default_domain,
        use_story_concatenation=False,
        tracker_limit=1000,
        remove_duplicates=False,
    )
    existing_stories = {t.export_stories() for t in existing_trackers}
    for t in recovered_trackers:
        story_str = t.export_stories()
        assert story_str in existing_stories
        existing_stories.discard(story_str)


async def test_persist_form_story(tmpdir):
    domain = Domain.load("data/test_domains/form.yml")

    tracker = DialogueStateTracker("", domain.slots)

    story = (
        "* greet\n"
        "    - utter_greet\n"
        "* start_form\n"
        "    - some_form\n"
        '    - form{"name": "some_form"}\n'
        "* default\n"
        "    - utter_default\n"
        "    - some_form\n"
        "* stop\n"
        "    - utter_ask_continue\n"
        "* affirm\n"
        "    - some_form\n"
        "* stop\n"
        "    - utter_ask_continue\n"
        "    - action_listen\n"
        "* form: inform\n"
        "    - some_form\n"
        '    - form{"name": null}\n'
        "* goodbye\n"
        "    - utter_goodbye\n"
    )

    # simulate talking to the form
    events = [
        UserUttered(intent={"name": "greet"}),
        ActionExecuted("utter_greet"),
        ActionExecuted("action_listen"),
        # start the form
        UserUttered(intent={"name": "start_form"}),
        ActionExecuted("some_form"),
        Form("some_form"),
        ActionExecuted("action_listen"),
        # out of form input
        UserUttered(intent={"name": "default"}),
        ActionExecutionRejected("some_form"),
        ActionExecuted("utter_default"),
        ActionExecuted("some_form"),
        ActionExecuted("action_listen"),
        # out of form input
        UserUttered(intent={"name": "stop"}),
        ActionExecutionRejected("some_form"),
        ActionExecuted("utter_ask_continue"),
        ActionExecuted("action_listen"),
        # out of form input but continue with the form
        UserUttered(intent={"name": "affirm"}),
        FormValidation(False),
        ActionExecuted("some_form"),
        ActionExecuted("action_listen"),
        # out of form input
        UserUttered(intent={"name": "stop"}),
        ActionExecutionRejected("some_form"),
        ActionExecuted("utter_ask_continue"),
        ActionExecuted("action_listen"),
        # form input
        UserUttered(intent={"name": "inform"}),
        FormValidation(True),
        ActionExecuted("some_form"),
        ActionExecuted("action_listen"),
        Form(None),
        UserUttered(intent={"name": "goodbye"}),
        ActionExecuted("utter_goodbye"),
        ActionExecuted("action_listen"),
    ]
    [tracker.update(e) for e in events]

    assert story in tracker.export_stories()


async def test_read_stories_with_multiline_comments(tmpdir, default_domain):
    reader = MarkdownStoryReader(RegexInterpreter(), default_domain)

    story_steps = await reader.read_from_file(
        "data/test_stories/stories_with_multiline_comments.md"
    )

    assert len(story_steps) == 4
    assert story_steps[0].block_name == "happy path"
    assert len(story_steps[0].events) == 4
    assert story_steps[1].block_name == "sad path 1"
    assert len(story_steps[1].events) == 7
    assert story_steps[2].block_name == "sad path 2"
    assert len(story_steps[2].events) == 7
    assert story_steps[3].block_name == "say goodbye"
    assert len(story_steps[3].events) == 2
