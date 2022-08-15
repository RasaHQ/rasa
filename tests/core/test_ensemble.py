from rasa.shared.exceptions import InvalidConfigException
import pytest
import itertools
from typing import List, Tuple

from rasa.engine.graph import ExecutionContext
from rasa.engine.storage.resource import Resource
from rasa.engine.storage.storage import ModelStorage
from rasa.core.policies.policy import PolicyPrediction
from rasa.core.policies.ensemble import DefaultPolicyPredictionEnsemble
from rasa.shared.core.domain import Domain
from rasa.shared.core.trackers import DialogueStateTracker
from rasa.shared.core.events import ActionExecutionRejected, UserUttered
from rasa.shared.core.events import ActionExecuted, DefinePrevUserUtteredFeaturization
from rasa.shared.core.constants import ACTION_LISTEN_NAME


@pytest.fixture
def default_ensemble(
    default_model_storage: ModelStorage, default_execution_context: ExecutionContext
) -> DefaultPolicyPredictionEnsemble:
    return DefaultPolicyPredictionEnsemble.create(
        config=DefaultPolicyPredictionEnsemble.get_default_config(),
        model_storage=default_model_storage,
        resource=Resource("ensemble"),
        execution_context=default_execution_context,
    )


def test_default_predict_complains_if_no_predictions_given(
    default_ensemble: DefaultPolicyPredictionEnsemble,
):
    domain = Domain.load("data/test_domains/default.yml")
    tracker = DialogueStateTracker.from_events(sender_id="arbitrary", evts=[])
    with pytest.raises(InvalidConfigException):
        default_ensemble.combine_predictions_from_kwargs(domain=domain, tracker=tracker)


def test_default_predict_ignores_other_kwargs(
    default_ensemble: DefaultPolicyPredictionEnsemble,
):
    domain = Domain.load("data/test_domains/default.yml")
    tracker = DialogueStateTracker.from_events(sender_id="arbitrary", evts=[])
    prediction = PolicyPrediction(
        policy_name="arbitrary", probabilities=[1.0], policy_priority=1
    )

    final_prediction = default_ensemble.combine_predictions_from_kwargs(
        domain=domain,
        tracker=tracker,
        **{
            "policy-graph-component-1": prediction,
            "another-random-component": domain,
            "yet-another-component": tracker,
        },
    )
    assert final_prediction.policy_name == prediction.policy_name


def test_default_predict_excludes_rejected_action(
    default_ensemble: DefaultPolicyPredictionEnsemble,
):
    domain = Domain.load("data/test_domains/default.yml")
    excluded_action = domain.action_names_or_texts[0]
    tracker = DialogueStateTracker.from_events(
        sender_id="arbitrary",
        evts=[
            UserUttered("hi"),
            ActionExecuted(excluded_action),
            ActionExecutionRejected(excluded_action),  # not "Rejection"
        ],
    )
    num_actions = len(domain.action_names_or_texts)
    predictions = [
        PolicyPrediction(
            policy_name=str(idx), probabilities=[1.0] * num_actions, policy_priority=idx
        )
        for idx in range(2)
    ]
    index_of_excluded_action = domain.index_for_action(excluded_action)
    prediction = default_ensemble.combine_predictions_from_kwargs(
        domain=domain,
        tracker=tracker,
        **{prediction.policy_name: prediction for prediction in predictions},
    )
    assert prediction.probabilities[index_of_excluded_action] == 0.0


@pytest.mark.parametrize(
    "predictions_and_expected_winner_idx, last_action_was_action_listen",
    itertools.product(
        [
            (
                # highest probability and highest priority
                [
                    PolicyPrediction(
                        policy_name=str(idx),
                        probabilities=[idx] * 3,
                        policy_priority=idx,
                    )
                    for idx in range(4)
                ],
                3,
            ),
            (
                # highest probability wins even if priority is low
                [
                    PolicyPrediction(
                        policy_name=str(idx),
                        probabilities=[idx] * 3,
                        policy_priority=idx,
                    )
                    for idx in reversed(range(4))
                ],
                0,
            ),
            (
                # "end to end" prediction supersedes others
                [
                    PolicyPrediction(
                        policy_name="policy using user text but max prob 0.0 wins",
                        probabilities=[0.0],
                        policy_priority=0,
                        is_end_to_end_prediction=True,
                    ),
                    PolicyPrediction(
                        policy_name="policy not using user text but max prob 1.0",
                        probabilities=[1.0],
                        policy_priority=1,
                        is_end_to_end_prediction=False,
                    ),
                ],
                0,
            ),
            (
                # "no user" prediction supsersedes even the end to end ones
                [
                    PolicyPrediction(
                        policy_name="'no user' with smallest max. prob",
                        probabilities=[0.0],
                        policy_priority=0,
                        is_no_user_prediction=True,
                    ),
                    PolicyPrediction(
                        policy_name="'end2end' with higher prob and priority",
                        probabilities=[1.0],
                        policy_priority=1,
                        is_end_to_end_prediction=True,
                    ),
                    PolicyPrediction(
                        policy_name="highest prob and highest priority",
                        probabilities=[2.0],
                        policy_priority=2,
                    ),
                ],
                0,
            ),
        ],
        [True, False],
    ),
)
def test_default_combine_predictions(
    default_ensemble: DefaultPolicyPredictionEnsemble,
    predictions_and_expected_winner_idx: Tuple[List[PolicyPrediction], int],
    last_action_was_action_listen: bool,
):
    domain = Domain.load("data/test_domains/default.yml")
    predictions, expected_winner_idx = predictions_and_expected_winner_idx

    # add mandatory and optional events to every prediction
    for prediction in predictions:
        prediction.events = [ActionExecuted(action_name=prediction.policy_name)]
        prediction.optional_events = [
            ActionExecuted(action_name=f"optional-{prediction.policy_name}")
        ]

    # expected events
    expected_events = set(
        event for prediction in predictions for event in prediction.events
    )
    expected_events.update(predictions[expected_winner_idx].optional_events)
    if last_action_was_action_listen:
        expected_events.add(
            DefinePrevUserUtteredFeaturization(
                predictions[expected_winner_idx].is_end_to_end_prediction
            )
        )

    # construct tracker
    evts = (
        [ActionExecuted(action_name=ACTION_LISTEN_NAME)]
        if last_action_was_action_listen
        else []
    )
    tracker = DialogueStateTracker.from_events(sender_id="arbitrary", evts=evts)

    # get the best prediction!
    best_prediction = default_ensemble.combine_predictions_from_kwargs(
        tracker,
        domain=domain,
        **{prediction.policy_name: prediction for prediction in predictions},
    )

    # compare events first ...
    assert set(best_prediction.events) == expected_events
    assert not best_prediction.optional_events

    # ... then drop events and compare the rest
    best_prediction.events = []
    best_prediction.optional_events = []
    predictions[expected_winner_idx].events = []
    predictions[expected_winner_idx].optional_events = []

    # ... not quite there yet, because old implementation creates a policy with
    # best_policy.priority as priority and the first one is a tuple which then
    # becomes a tuple with a tuple with an int, so...
    predictions[expected_winner_idx].policy_priority = predictions[
        expected_winner_idx
    ].policy_priority

    # now, we can compare:
    assert best_prediction == predictions[expected_winner_idx]
