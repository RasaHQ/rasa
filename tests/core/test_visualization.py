from rasa.core.events import ActionExecuted, SlotSet, UserUttered
from rasa.core.training import visualization


def test_style_transfer():
    r = visualization._transfer_style({"class": "dashed great"}, {"class": "myclass"})
    assert r["class"] == "myclass dashed"


def test_style_transfer_empty():
    r = visualization._transfer_style({"class": "dashed great"}, {"something": "else"})
    assert r["class"] == "dashed"


def test_common_action_prefix():
    this = [
        ActionExecuted("action_listen"),
        ActionExecuted("greet"),
        UserUttered("hey"),
        ActionExecuted("amazing"),
        # until this point they are the same
        SlotSet("my_slot", "a"),
        ActionExecuted("a"),
        ActionExecuted("after_a"),
    ]
    other = [
        ActionExecuted("action_listen"),
        ActionExecuted("greet"),
        UserUttered("hey"),
        ActionExecuted("amazing"),
        # until this point they are the same
        SlotSet("my_slot", "b"),
        ActionExecuted("b"),
        ActionExecuted("after_b"),
    ]
    num_common = visualization._length_of_common_action_prefix(this, other)

    assert num_common == 3


def test_common_action_prefix_equal():
    this = [
        ActionExecuted("action_listen"),
        ActionExecuted("greet"),
        UserUttered("hey"),
        ActionExecuted("amazing"),
    ]
    other = [
        ActionExecuted("action_listen"),
        ActionExecuted("greet"),
        UserUttered("hey"),
        ActionExecuted("amazing"),
    ]
    num_common = visualization._length_of_common_action_prefix(this, other)

    assert num_common == 3


def test_common_action_prefix_unequal():
    this = [
        ActionExecuted("action_listen"),
        ActionExecuted("greet"),
        UserUttered("hey"),
    ]
    other = [
        ActionExecuted("greet"),
        ActionExecuted("action_listen"),
        UserUttered("hey"),
    ]
    num_common = visualization._length_of_common_action_prefix(this, other)

    assert num_common == 0


async def test_graph_persistence(default_domain, tmpdir):
    from os.path import isfile
    from networkx.drawing import nx_pydot
    from rasa.core.training.dsl import StoryFileReader
    from rasa.core.interpreter import RegexInterpreter

    story_steps = await StoryFileReader.read_from_file(
        "data/test_stories/stories.md", default_domain, interpreter=RegexInterpreter()
    )
    out_file = tmpdir.join("graph.html").strpath
    generated_graph = await visualization.visualize_stories(
        story_steps,
        default_domain,
        output_file=out_file,
        max_history=3,
        should_merge_nodes=False,
    )

    generated_graph = nx_pydot.to_pydot(generated_graph)

    assert isfile(out_file)

    with open(out_file, "r") as graph_file:
        content = graph_file.read()

    assert "isClient = true" in content
    assert "graph = `{}`".format(generated_graph.to_string()) in content
