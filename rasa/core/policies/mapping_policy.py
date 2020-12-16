import logging
from typing import Any, List, Text, Optional, Dict, TYPE_CHECKING

import rasa.shared.utils.common
import rasa.utils.io
import rasa.shared.utils.io
from rasa.shared.constants import DOCS_URL_POLICIES, DOCS_URL_MIGRATION_GUIDE
from rasa.shared.nlu.constants import INTENT_NAME_KEY
from rasa.shared.core.constants import (
    USER_INTENT_BACK,
    USER_INTENT_RESTART,
    USER_INTENT_SESSION_START,
    ACTION_LISTEN_NAME,
    ACTION_RESTART_NAME,
    ACTION_SESSION_START_NAME,
    ACTION_BACK_NAME,
)
from rasa.shared.core.domain import InvalidDomain, Domain
from rasa.shared.core.events import ActionExecuted
from rasa.shared.nlu.interpreter import NaturalLanguageInterpreter
from rasa.core.policies.policy import Policy, PolicyPrediction
from rasa.shared.core.trackers import DialogueStateTracker
from rasa.shared.core.generator import TrackerWithCachedStates
from rasa.core.constants import MAPPING_POLICY_PRIORITY

if TYPE_CHECKING:
    from rasa.core.policies.ensemble import PolicyEnsemble


logger = logging.getLogger(__name__)


class MappingPolicy(Policy):
    """Policy which maps intents directly to actions.

    Intents can be assigned actions in the domain file which are to be
    executed whenever the intent is detected. This policy takes precedence over
    any other policy.
    """

    @staticmethod
    def _standard_featurizer() -> None:
        return None

    def __init__(self, priority: int = MAPPING_POLICY_PRIORITY, **kwargs: Any) -> None:
        """Create a new Mapping policy."""
        super().__init__(priority=priority, **kwargs)

        rasa.shared.utils.io.raise_deprecation_warning(
            f"'{MappingPolicy.__name__}' is deprecated and will be removed in "
            "the future. It is recommended to use the 'RulePolicy' instead.",
            docs=DOCS_URL_MIGRATION_GUIDE,
        )

    @classmethod
    def validate_against_domain(
        cls, ensemble: Optional["PolicyEnsemble"], domain: Optional[Domain]
    ) -> None:
        if not domain:
            return

        has_mapping_policy = ensemble is not None and any(
            isinstance(p, cls) for p in ensemble.policies
        )
        has_triggers_in_domain = any(
            [
                "triggers" in properties
                for intent, properties in domain.intent_properties.items()
            ]
        )
        if has_triggers_in_domain and not has_mapping_policy:
            raise InvalidDomain(
                "You have defined triggers in your domain, but haven't "
                "added the MappingPolicy to your policy ensemble. "
                "Either remove the triggers from your domain or "
                "include the MappingPolicy in your policy configuration."
            )

    def train(
        self,
        training_trackers: List[TrackerWithCachedStates],
        domain: Domain,
        interpreter: NaturalLanguageInterpreter,
        **kwargs: Any,
    ) -> None:
        """Does nothing. This policy is deterministic."""

        pass

    def predict_action_probabilities(
        self,
        tracker: DialogueStateTracker,
        domain: Domain,
        interpreter: NaturalLanguageInterpreter,
        **kwargs: Any,
    ) -> PolicyPrediction:
        """Predicts the assigned action.

        If the current intent is assigned to an action that action will be
        predicted with the highest probability of all policies. If it is not
        the policy will predict zero for every action.
        """
        result = self._default_predictions(domain)

        intent = tracker.latest_message.intent.get(INTENT_NAME_KEY)
        if intent == USER_INTENT_RESTART:
            action = ACTION_RESTART_NAME
        elif intent == USER_INTENT_BACK:
            action = ACTION_BACK_NAME
        elif intent == USER_INTENT_SESSION_START:
            action = ACTION_SESSION_START_NAME
        else:
            action = domain.intent_properties.get(intent, {}).get("triggers")

        if tracker.latest_action_name == ACTION_LISTEN_NAME:
            # predict mapped action
            if action:
                idx = domain.index_for_action(action)
                if idx is None:
                    rasa.shared.utils.io.raise_warning(
                        f"MappingPolicy tried to predict unknown "
                        f"action '{action}'. Make sure all mapped actions are "
                        f"listed in the domain.",
                        docs=DOCS_URL_POLICIES + "#mapping-policy",
                    )
                else:
                    result[idx] = 1

            if any(result):
                logger.debug(
                    "The predicted intent '{}' is mapped to "
                    " action '{}' in the domain."
                    "".format(intent, action)
                )
        elif tracker.latest_action_name == action and action is not None:
            # predict next action_listen after mapped action
            latest_action = tracker.get_last_event_for(ActionExecuted)
            assert latest_action.action_name == action
            if latest_action.policy and latest_action.policy.endswith(
                type(self).__name__
            ):
                # this ensures that we only predict listen,
                # if we predicted the mapped action
                logger.debug(
                    "The mapped action, '{}', for this intent, '{}', was "
                    "executed last so MappingPolicy is returning to "
                    "action_listen.".format(action, intent)
                )

                idx = domain.index_for_action(ACTION_LISTEN_NAME)
                result[idx] = 1
            else:
                logger.debug(
                    "The mapped action, '{}', for the intent, '{}', was "
                    "executed last, but it was predicted by another policy, '{}', "
                    "so MappingPolicy is not predicting any action.".format(
                        action, intent, latest_action.policy
                    )
                )
        elif action == ACTION_RESTART_NAME:
            logger.debug("Restarting the conversation with action_restart.")
            idx = domain.index_for_action(ACTION_RESTART_NAME)
            result[idx] = 1
        else:
            logger.debug(
                "There is no mapped action for the predicted intent, "
                "'{}'.".format(intent)
            )
        return self._prediction(result)

    def _metadata(self) -> Dict[Text, Any]:
        return {"priority": self.priority}

    @classmethod
    def _metadata_filename(cls) -> Text:
        return "mapping_policy.json"
