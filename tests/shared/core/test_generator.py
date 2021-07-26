import rasa.shared.core.generator
from rasa.shared.core.domain import Domain
from rasa.shared.core.events import UserUttered, ActionExecuted
from rasa.shared.core.slots import TextSlot


def test_subsample_array_read_only():
    t = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    r = rasa.shared.core.generator._subsample_array(
        t, 5, can_modify_incoming_array=False
    )

    assert len(r) == 5
    assert set(r).issubset(t)


def test_subsample_array():
    t = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    # this will modify the original array and shuffle it
    r = rasa.shared.core.generator._subsample_array(t, 5)

    assert len(r) == 5
    assert set(r).issubset(t)


def test_tracker_with_cached_states_fingerprinting_consistency(domain: Domain):
    slot = TextSlot(name="name", influence_conversation=True)
    slot.value = "example"
    tr1 = rasa.shared.core.generator.TrackerWithCachedStates(
        "test_sender_id", slots=[slot], domain=domain
    )
    tr2 = rasa.shared.core.generator.TrackerWithCachedStates(
        "test_sender_id", slots=[slot], domain=domain
    )
    f1 = tr1.fingerprint()
    f2 = tr2.fingerprint()
    assert f1 == f2


def test_tracker_with_cached_states_unique_fingerprint(domain: Domain):
    slot = TextSlot(name="name", influence_conversation=True)
    slot.value = "example"
    tr = rasa.shared.core.generator.TrackerWithCachedStates(
        "test_sender_id", slots=[slot], domain=domain
    )
    f1 = tr.fingerprint()

    event1 = UserUttered(
        text="hello",
        parse_data={
            "intent": {"id": 2, "name": "greet", "confidence": 0.9604260921478271},
            "entities": [
                {"entity": "city", "value": "London"},
                {"entity": "count", "value": 1},
            ],
            "text": "hi",
            "message_id": "3f4c04602a4947098c574b107d3ccc59",
            "metadata": {},
            "intent_ranking": [
                {"id": 2, "name": "greet", "confidence": 0.9604260921478271},
                {"id": 1, "name": "goodbye", "confidence": 0.01835782080888748},
                {"id": 0, "name": "deny", "confidence": 0.011255578137934208},
            ],
        },
    )
    tr.update(event1)
    f2 = tr.fingerprint()
    assert f1 != f2

    event2 = ActionExecuted(action_name="action_listen")
    tr.update(event2)
    f3 = tr.fingerprint()
    assert f2 != f3
