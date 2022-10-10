from rasa.core import training
from rasa.shared.core.domain import Domain


async def test_generated_trackers_can_omit_unset_slots():
    domain = Domain.from_path(
        "data/test_domains/initial_slot_values_greet_and_goodbye.yml"
    )

    trackers = await training.load_data(
        "data/test_yaml_stories/rules_greet_and_goodbye.yml", domain
    )

    assert len(trackers) == 2
    assert all([t.is_rule_tracker for t in trackers])

    states_without_unset_slots = trackers[0].past_states(domain, omit_unset_slots=True)
    assert not any(["slots" in state for state in states_without_unset_slots])

    states_with_unset_slots = trackers[0].past_states(domain, omit_unset_slots=False)
    assert all(["slots" in state for state in states_with_unset_slots])
