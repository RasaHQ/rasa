from typing import Text, List, Optional, Union, TYPE_CHECKING, Dict, Set
from collections import defaultdict

if TYPE_CHECKING:
    from rasa.shared.core.domain import Domain
    from rasa.shared.core.trackers import DialogueStateTracker
    from rasa.shared.core.generator import TrackerWithCachedStates
    from rasa.shared.core.training_data.structures import StoryGraph
    from rasa.shared.importers.importer import TrainingDataImporter
    from rasa.shared.core.events import Event


async def extract_rule_data(
    resource_name: Text,
    domain: "Domain",
    use_e2e: bool = False,
    exclusion_percentage: int = None,
) -> "StoryGraph":
    from rasa.shared.core.training_data import loading
    from rasa.shared.core.training_data.structures import StoryGraph

    story_steps = await loading.load_data_from_resource(
        resource_name,
        domain,
        use_e2e=use_e2e,
        exclusion_percentage=exclusion_percentage,
    )
    return StoryGraph(story_steps)


async def extract_story_graph(
    resource_name: Text,
    domain: "Domain",
    use_e2e: bool = False,
    exclusion_percentage: Optional[int] = None,
) -> "StoryGraph":
    from rasa.shared.core.training_data.structures import StoryGraph
    import rasa.shared.core.training_data.loading as core_loading

    story_steps = await core_loading.load_data_from_resource(
        resource_name,
        domain,
        use_e2e=use_e2e,
        exclusion_percentage=exclusion_percentage,
    )
    return StoryGraph(story_steps)


async def load_data(
    resource_name: Union[Text, "TrainingDataImporter"],
    domain: "Domain",
    remove_duplicates: bool = True,
    unique_last_num_states: Optional[int] = None,
    augmentation_factor: int = 50,
    tracker_limit: Optional[int] = None,
    use_story_concatenation: bool = True,
    debug_plots: bool = False,
    exclusion_percentage: Optional[int] = None,
) -> List["TrackerWithCachedStates"]:
    """
    Load training data from a resource.

    Args:
        resource_name: resource to load the data from. either a path or an importer
        domain: domain used for loading
        remove_duplicates: should duplicated training examples be removed?
        unique_last_num_states: number of states in a conversation that make the
            a tracker unique (this is used to identify duplicates)
        augmentation_factor:
            by how much should the story training data be augmented
        tracker_limit:
            maximum number of trackers to generate during augmentation
        use_story_concatenation:
            should stories be concatenated when doing data augmentation
        debug_plots:
            generate debug plots during loading
        exclusion_percentage:
            how much data to exclude

    Returns:
        list of loaded trackers
    """
    from rasa.shared.core.generator import TrainingDataGenerator
    from rasa.shared.importers.importer import TrainingDataImporter

    if resource_name:
        if isinstance(resource_name, TrainingDataImporter):
            graph = await resource_name.get_stories(
                exclusion_percentage=exclusion_percentage
            )
        else:
            graph = await extract_story_graph(
                resource_name, domain, exclusion_percentage=exclusion_percentage
            )

        g = TrainingDataGenerator(
            graph,
            domain,
            remove_duplicates,
            unique_last_num_states,
            augmentation_factor,
            tracker_limit,
            use_story_concatenation,
            debug_plots,
        )
        return g.generate()
    else:
        return []


def persist_data(trackers: List["DialogueStateTracker"], path: Text) -> None:
    """Dump a list of dialogue trackers in the story format to disk."""

    for t in trackers:
        t.export_stories_to_file(path)


def _find_events_after_actions(
    trackers: List["DialogueStateTracker"],
) -> Dict[Text, Set["Event"]]:
    """Creates a dictionary of action names and events that follow these actions.

    Args:
        trackers: the list of trackers

    Returns:
        a dictionary of action names and events that follow these actions
    """
    from rasa.shared.core.events import ActionExecuted

    events_after_actions = defaultdict(set)

    for t in trackers:
        tracker = t.init_copy()
        for event in t.events:
            tracker.update(event)
            if isinstance(event, ActionExecuted):
                continue

            action_name = tracker.latest_action_name
            if action_name:
                events_after_actions[action_name].add(event)

    return events_after_actions


def create_action_fingerprints(
    trackers: List["DialogueStateTracker"], domain: "Domain"
) -> Dict[Text, Dict[Text, List[Text]]]:
    """Fingerprint each action using the events it created during train.

    This allows us to emit warnings when the model is used
    if an action does things it hasn't done during training,
    or if rules are incomplete.

    Args:
        trackers: the list of trackers
        domain: the domain

    Returns:
        a nested dictionary of action names and slots and active loops
            that this action sets
    """
    from rasa.shared.core.events import (
        SlotSet,
        ActiveLoop,
    )
    from rasa.shared.core.constants import (
        SLOTS,
        ACTIVE_LOOP,
    )

    events_after_actions = _find_events_after_actions(trackers)
    if not events_after_actions:
        return {}

    # take into account only featurized slots
    featurized_slots = {slot.name for slot in domain.slots if slot.has_features()}
    action_fingerprints = defaultdict(dict)
    for action_name, events_after_action in events_after_actions.items():
        slots = list(
            set(
                event.key for event in events_after_action if isinstance(event, SlotSet)
            ).intersection(featurized_slots)
        )
        active_loops = list(
            set(
                event.name
                for event in events_after_action
                if isinstance(event, ActiveLoop)
            )
        )

        if slots:
            action_fingerprints[action_name][SLOTS] = slots
        if active_loops:
            action_fingerprints[action_name][ACTIVE_LOOP] = active_loops

    return action_fingerprints
