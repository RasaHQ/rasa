from pathlib import Path
from typing import Text

import pytest

from rasa.core.domain import Domain
from rasa.core.events import ActionExecuted, SlotSet, UserUttered
from rasa.core.training import visualization
import rasa.utils.io


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


@pytest.mark.parametrize(
    "stories_file",
    ["data/test_stories/stories.md", "data/test_yaml_stories/stories.yml"],
)
async def test_graph_persistence(
    stories_file: Text, default_domain: Domain, tmp_path: Path
):
    from os.path import isfile
    from networkx.drawing import nx_pydot
    import rasa.core.training.loading as core_loading

    story_steps = await core_loading.load_data_from_resource(
        stories_file, default_domain
    )
    out_file = str(tmp_path / "graph.html")
    generated_graph = await visualization.visualize_stories(
        story_steps,
        default_domain,
        output_file=out_file,
        max_history=3,
        should_merge_nodes=False,
    )

    generated_graph = nx_pydot.to_pydot(generated_graph)

    assert isfile(out_file)

    content = rasa.utils.io.read_file(out_file)

    assert "isClient = true" in content
    assert "graph = `{}`".format(generated_graph.to_string()) in content


@pytest.mark.parametrize(
    "stories_file",
    ["data/test_stories/stories.md", "data/test_yaml_stories/stories.yml"],
)
async def test_merge_nodes(stories_file: Text, default_domain: Domain, tmp_path: Path):
    from os.path import isfile
    import rasa.core.training.loading as core_loading

    story_steps = await core_loading.load_data_from_resource(
        stories_file, default_domain
    )
    out_file = str(tmp_path / "graph.html")
    await visualization.visualize_stories(
        story_steps,
        default_domain,
        output_file=out_file,
        max_history=3,
        should_merge_nodes=True,
    )
    assert isfile(out_file)
