import logging
import os
import collections
import pathlib
from typing import Text, List, Tuple, DefaultDict, Generator
import json

import rasa.shared.core.events
from rasa import telemetry
from rasa.shared.core.training_data import loading
from rasa.shared.utils.cli import print_error
from rasa.shared.core.domain import InvalidDomain, Domain
from rasa.shared.core.training_data.structures import StoryStep
from rasa.shared.core.events import UserUttered, ActionExecuted

logger = logging.getLogger(__name__)


def visualize(
    domain_path: Text,
    stories_path: Text,
    nlu_data_path: Text,
    output_path: Text,
    max_history: int,
) -> None:
    """Visualizes stories as graph.

    Args:
        domain_path: Path to the domain file.
        stories_path: Path to the stories files.
        nlu_data_path: Path to the NLU training data which can be used to interpolate
            intents with actual examples in the graph.
        output_path: Path where the created graph should be persisted.
        max_history: Max history to use for the story visualization.
    """
    import rasa.shared.core.training_data.visualization

    try:
        domain = Domain.load(domain_path)
    except InvalidDomain as e:
        print_error(
            f"Could not load domain due to: '{e}'. To specify a valid domain path use "
            f"the '--domain' argument."
        )
        return

    # this is optional, only needed if the `/greet` type of
    # messages in the stories should be replaced with actual
    # messages (e.g. `hello`)
    if nlu_data_path is not None:
        import rasa.shared.nlu.training_data.loading

        nlu_training_data = rasa.shared.nlu.training_data.loading.load_data(
            nlu_data_path
        )
    else:
        nlu_training_data = None

    logger.info("Starting to visualize stories...")
    telemetry.track_visualization()

    story_steps = loading.load_data_from_resource(stories_path, domain)
    rasa.shared.core.training_data.visualization.visualize_stories(
        story_steps,
        domain,
        output_path,
        max_history,
        nlu_training_data=nlu_training_data,
    )

    full_output_path = "file://{}".format(os.path.abspath(output_path))
    logger.info(f"Finished graph creation. Saved into {full_output_path}")

    import webbrowser

    webbrowser.open(full_output_path)


def iterate_step_intents_and_actions(step: StoryStep) -> Generator[Text, None, None]:
    for event in step.events:
        if isinstance(event, UserUttered):
            if event.intent_name is not None:
                yield event.intent_name
        elif isinstance(event, ActionExecuted):
            yield event.action_name


def story_steps_to_paths(story_steps: List[StoryStep]) -> List[List[Text]]:
    """
    Traverse story graph into all paths.
    :param story_steps: list of story steps
    :return: list of all paths (including intents and actions)
    """
    # convert all story steps into (start, end, paths) chunks
    path_chunks = []
    # map start checkpoint name to end checkpoint name + chunk index
    start_idx: DefaultDict[Text, Tuple[Optional[Text], int]] = collections.defaultdict(list)

    for step in story_steps:
        chunk = list(iterate_step_intents_and_actions(step))
        if not chunk:
            continue

        chunk_idx = len(path_chunks)
        path_chunks.append(chunk)

        end_names = [chpt.name for chpt in step.end_checkpoints]
        # end of story is empty list of checkpoints
        if not end_names:
            end_names.append(None)
        for start_name in map(lambda chpt: chpt.name, step.start_checkpoints):
            for end_name in end_names:
                start_idx[start_name].append((end_name, chunk_idx))

    deque = collections.deque([('STORY_START', [])])
    result = []
    while deque:
        end_name, chunk = deque.pop()
        for cont_name, chunk_idx in start_idx[end_name]:
            new_chunk = list(chunk)
            new_chunk.extend(path_chunks[chunk_idx])
            if cont_name is None:
                # end of chunk, append to the result
                result.append(new_chunk)
            else:
                # multi-step checkpoint, push to the queue
                deque.append((cont_name, new_chunk))
    return result


def story_steps_to_file_groups(story_steps: List[StoryStep]) -> DefaultDict[Text, List[Text]]:
    """
    Group intents and actions by file names they were loaded from.
    :param story_steps: list of story steps to process
    :return: dict, mapping filename to list of intents and actions
    """
    res = collections.defaultdict(list)
    for step in story_steps:
        file_name = pathlib.Path(step.source_name).name
        l = res[file_name]
        l.extend(iterate_step_intents_and_actions(step))
        res[file_name] = list(set(l))
    return res


def file_groups_normalize_names(file_groups: DefaultDict[Text, List[Text]]) -> DefaultDict[Text, List[Text]]:
    res = collections.defaultdict(list)
    name_indices = collections.Counter()
    for file_name, group in file_groups.items():
        _, fname = file_name.split('_', maxsplit=1)
        idx = name_indices[fname]
        if idx == 0:
            res_name = fname
        else:
            res_name = f"{fname}-{idx}"
        name_indices[fname] += 1
        res[res_name] = group
    return res


def dump_graph(domain_path: Text, stories_path: Text, nlu_data_path: Text, output_path: Text):
    logger.info("Loading domain and stories data...")
    try:
        domain = Domain.load(domain_path)
    except InvalidDomain as e:
        print_error(f"Could not load domain due to: '{e}'")
        return
    story_steps = loading.load_data_from_resource(stories_path, domain)
    logger.info("Data loaded, generating graph...")
    stories_paths = story_steps_to_paths(story_steps)
    file_groups = story_steps_to_file_groups(story_steps)
    file_groups = file_groups_normalize_names(file_groups)

    # add fallback actions from intents
    for intent, intent_prop in domain.intent_properties.items():
        action = intent_prop.get('fallback_action')
        if not action:
            continue
        stories_paths.append([intent, action])

    # add buttons in resposnes
    for resp, resp_props in domain.responses.items():
        for prop in resp_props:
            for button in prop.get('buttons', []):
                action = button.get('payload', "").lstrip('/')
                if action:
                    stories_paths.append([resp, action])

    data = {
        'intents': domain.intents,
        'actions': domain.action_names_or_texts,
        'stories_paths': stories_paths,
        'file_groups': file_groups,
    }

    with open(output_path, "wt", encoding='utf-8') as fd:
        json.dump(data, fd, indent=4)