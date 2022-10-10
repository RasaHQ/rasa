import logging
import os
import warnings as pywarnings
import typing
from collections import defaultdict, namedtuple
from typing import Any, Dict, List, Optional, Text, Tuple

from rasa import telemetry
from rasa.core.constants import (
    CONFUSION_MATRIX_STORIES_FILE,
    REPORT_STORIES_FILE,
    FAILED_STORIES_FILE,
    SUCCESSFUL_STORIES_FILE,
    STORIES_WITH_WARNINGS_FILE,
)
from rasa.core.policies.policy import PolicyPrediction
from rasa.nlu.test import EntityEvaluationResult, evaluate_entities
from rasa.shared.core.constants import (
    POLICIES_THAT_EXTRACT_ENTITIES,
    ACTION_UNLIKELY_INTENT_NAME,
)
from rasa.shared.exceptions import RasaException
from rasa.shared.nlu.training_data.message import Message
import rasa.shared.utils.io
from rasa.core.channels import UserMessage
from rasa.shared.core.training_data.story_writer.yaml_story_writer import (
    YAMLStoryWriter,
)
from rasa.shared.core.training_data.structures import StoryStep
from rasa.shared.core.domain import Domain
from rasa.nlu.constants import (
    RESPONSE_SELECTOR_DEFAULT_INTENT,
    RESPONSE_SELECTOR_RETRIEVAL_INTENTS,
    TOKENS_NAMES,
)
from rasa.shared.nlu.constants import (
    INTENT,
    ENTITIES,
    ENTITY_ATTRIBUTE_VALUE,
    ENTITY_ATTRIBUTE_START,
    ENTITY_ATTRIBUTE_END,
    EXTRACTOR,
    ENTITY_ATTRIBUTE_TYPE,
    INTENT_RESPONSE_KEY,
    INTENT_NAME_KEY,
    RESPONSE,
    RESPONSE_SELECTOR,
    FULL_RETRIEVAL_INTENT_NAME_KEY,
    TEXT,
    ENTITY_ATTRIBUTE_TEXT,
)
from rasa.constants import RESULTS_FILE, PERCENTAGE_KEY
from rasa.shared.core.events import ActionExecuted, EntitiesAdded, UserUttered
from rasa.shared.core.trackers import DialogueStateTracker
from rasa.shared.nlu.training_data.formats.readerwriter import TrainingDataWriter
from rasa.shared.importers.importer import TrainingDataImporter
from rasa.shared.utils.io import DEFAULT_ENCODING
from rasa.utils.tensorflow.constants import QUERY_INTENT_KEY, SEVERITY_KEY

if typing.TYPE_CHECKING:
    from rasa.core.agent import Agent
    from rasa.core.processor import MessageProcessor
    from rasa.shared.core.generator import TrainingDataGenerator
    from rasa.shared.core.events import EntityPrediction

logger = logging.getLogger(__name__)

StoryEvaluation = namedtuple(
    "StoryEvaluation",
    [
        "evaluation_store",
        "failed_stories",
        "successful_stories",
        "stories_with_warnings",
        "action_list",
        "in_training_data_fraction",
    ],
)

PredictionList = List[Optional[Text]]


class WrongPredictionException(RasaException, ValueError):
    """Raised if a wrong prediction is encountered."""


class WarningPredictedAction(ActionExecuted):
    """The model predicted the correct action with warning."""

    type_name = "warning_predicted"

    def __init__(
        self,
        action_name_prediction: Text,
        action_name: Optional[Text] = None,
        policy: Optional[Text] = None,
        confidence: Optional[float] = None,
        timestamp: Optional[float] = None,
        metadata: Optional[Dict] = None,
    ):
        """Creates event `action_unlikely_intent` predicted as warning.

        See the docstring of the parent class for more information.
        """
        self.action_name_prediction = action_name_prediction
        super().__init__(action_name, policy, confidence, timestamp, metadata)

    def inline_comment(self) -> Text:
        """A comment attached to this event. Used during dumping."""
        return f"predicted: {self.action_name_prediction}"


class WronglyPredictedAction(ActionExecuted):
    """The model predicted the wrong action.

    Mostly used to mark wrong predictions and be able to
    dump them as stories.
    """

    type_name = "wrong_action"

    def __init__(
        self,
        action_name_target: Text,
        action_text_target: Text,
        action_name_prediction: Text,
        policy: Optional[Text] = None,
        confidence: Optional[float] = None,
        timestamp: Optional[float] = None,
        metadata: Optional[Dict] = None,
        predicted_action_unlikely_intent: bool = False,
    ) -> None:
        """Creates event for a successful event execution.

        See the docstring of the parent class `ActionExecuted` for more information.
        """
        self.action_name_prediction = action_name_prediction
        self.predicted_action_unlikely_intent = predicted_action_unlikely_intent
        super().__init__(
            action_name_target,
            policy,
            confidence,
            timestamp,
            metadata,
            action_text=action_text_target,
        )

    def inline_comment(self) -> Text:
        """A comment attached to this event. Used during dumping."""
        comment = f"predicted: {self.action_name_prediction}"
        if self.predicted_action_unlikely_intent:
            return f"{comment} after {ACTION_UNLIKELY_INTENT_NAME}"
        return comment

    def as_story_string(self) -> Text:
        """Returns the story equivalent representation."""
        return f"{self.action_name}   <!-- {self.inline_comment()} -->"

    def __repr__(self) -> Text:
        """Returns event as string for debugging."""
        return (
            f"WronglyPredictedAction(action_target: {self.action_name}, "
            f"action_prediction: {self.action_name_prediction}, "
            f"policy: {self.policy}, confidence: {self.confidence}, "
            f"metadata: {self.metadata})"
        )


class EvaluationStore:
    """Class storing action, intent and entity predictions and targets."""

    def __init__(
        self,
        action_predictions: Optional[PredictionList] = None,
        action_targets: Optional[PredictionList] = None,
        intent_predictions: Optional[PredictionList] = None,
        intent_targets: Optional[PredictionList] = None,
        entity_predictions: Optional[List["EntityPrediction"]] = None,
        entity_targets: Optional[List["EntityPrediction"]] = None,
    ) -> None:
        """Initialize store attributes."""
        self.action_predictions = action_predictions or []
        self.action_targets = action_targets or []
        self.intent_predictions = intent_predictions or []
        self.intent_targets = intent_targets or []
        self.entity_predictions: List["EntityPrediction"] = entity_predictions or []
        self.entity_targets: List["EntityPrediction"] = entity_targets or []

    def add_to_store(
        self,
        action_predictions: Optional[PredictionList] = None,
        action_targets: Optional[PredictionList] = None,
        intent_predictions: Optional[PredictionList] = None,
        intent_targets: Optional[PredictionList] = None,
        entity_predictions: Optional[List["EntityPrediction"]] = None,
        entity_targets: Optional[List["EntityPrediction"]] = None,
    ) -> None:
        """Add items or lists of items to the store."""
        self.action_predictions.extend(action_predictions or [])
        self.action_targets.extend(action_targets or [])
        self.intent_targets.extend(intent_targets or [])
        self.intent_predictions.extend(intent_predictions or [])
        self.entity_predictions.extend(entity_predictions or [])
        self.entity_targets.extend(entity_targets or [])

    def merge_store(self, other: "EvaluationStore") -> None:
        """Add the contents of other to self."""
        self.add_to_store(
            action_predictions=other.action_predictions,
            action_targets=other.action_targets,
            intent_predictions=other.intent_predictions,
            intent_targets=other.intent_targets,
            entity_predictions=other.entity_predictions,
            entity_targets=other.entity_targets,
        )

    def _check_entity_prediction_target_mismatch(self) -> bool:
        """Checks that same entities were expected and actually extracted.

        Possible duplicates or differences in order should not matter.
        """
        deduplicated_targets = set(
            tuple(entity.items()) for entity in self.entity_targets
        )
        deduplicated_predictions = set(
            tuple(entity.items()) for entity in self.entity_predictions
        )
        return deduplicated_targets != deduplicated_predictions

    def check_prediction_target_mismatch(self) -> bool:
        """Checks if intent, entity or action predictions don't match expected ones."""
        return (
            self.intent_predictions != self.intent_targets
            or self._check_entity_prediction_target_mismatch()
            or self.action_predictions != self.action_targets
        )

    @staticmethod
    def _compare_entities(
        entity_predictions: List["EntityPrediction"],
        entity_targets: List["EntityPrediction"],
        i_pred: int,
        i_target: int,
    ) -> int:
        """
        Compare the current predicted and target entities and decide which one
        comes first. If the predicted entity comes first it returns -1,
        while it returns 1 if the target entity comes first.
        If target and predicted are aligned it returns 0
        """
        pred = None
        target = None
        if i_pred < len(entity_predictions):
            pred = entity_predictions[i_pred]
        if i_target < len(entity_targets):
            target = entity_targets[i_target]
        if target and pred:
            # Check which entity has the lower "start" value
            if pred.get(ENTITY_ATTRIBUTE_START) < target.get(ENTITY_ATTRIBUTE_START):
                return -1
            elif target.get(ENTITY_ATTRIBUTE_START) < pred.get(ENTITY_ATTRIBUTE_START):
                return 1
            else:
                # Since both have the same "start" values,
                # check which one has the lower "end" value
                if pred.get(ENTITY_ATTRIBUTE_END) < target.get(ENTITY_ATTRIBUTE_END):
                    return -1
                elif target.get(ENTITY_ATTRIBUTE_END) < pred.get(ENTITY_ATTRIBUTE_END):
                    return 1
                else:
                    # The entities have the same "start" and "end" values
                    return 0
        return 1 if target else -1

    @staticmethod
    def _generate_entity_training_data(entity: Dict[Text, Any]) -> Text:
        return TrainingDataWriter.generate_entity(entity.get("text"), entity)

    def serialise(self) -> Tuple[PredictionList, PredictionList]:
        """Turn targets and predictions to lists of equal size for sklearn."""
        texts = sorted(
            set(
                [str(e.get("text", "")) for e in self.entity_targets]
                + [str(e.get("text", "")) for e in self.entity_predictions]
            )
        )

        aligned_entity_targets = []
        aligned_entity_predictions = []

        for text in texts:
            # sort the entities of this sentence to compare them directly
            entity_targets = sorted(
                filter(
                    lambda x: x.get(ENTITY_ATTRIBUTE_TEXT) == text, self.entity_targets
                ),
                key=lambda x: x.get(ENTITY_ATTRIBUTE_START),
            )
            entity_predictions = sorted(
                filter(
                    lambda x: x.get(ENTITY_ATTRIBUTE_TEXT) == text,
                    self.entity_predictions,
                ),
                key=lambda x: x.get(ENTITY_ATTRIBUTE_START),
            )

            i_pred, i_target = 0, 0

            while i_pred < len(entity_predictions) or i_target < len(entity_targets):
                cmp = self._compare_entities(
                    entity_predictions, entity_targets, i_pred, i_target
                )
                if cmp == -1:  # predicted comes first
                    aligned_entity_predictions.append(
                        self._generate_entity_training_data(entity_predictions[i_pred])
                    )
                    aligned_entity_targets.append("None")
                    i_pred += 1
                elif cmp == 1:  # target entity comes first
                    aligned_entity_targets.append(
                        self._generate_entity_training_data(entity_targets[i_target])
                    )
                    aligned_entity_predictions.append("None")
                    i_target += 1
                else:  # target and predicted entity are aligned
                    aligned_entity_predictions.append(
                        self._generate_entity_training_data(entity_predictions[i_pred])
                    )
                    aligned_entity_targets.append(
                        self._generate_entity_training_data(entity_targets[i_target])
                    )
                    i_pred += 1
                    i_target += 1

        targets = self.action_targets + self.intent_targets + aligned_entity_targets

        predictions = (
            self.action_predictions
            + self.intent_predictions
            + aligned_entity_predictions
        )
        return targets, predictions


class EndToEndUserUtterance(UserUttered):
    """End-to-end user utterance.

    Mostly used to print the full end-to-end user message in the
    `failed_test_stories.yml` output file.
    """

    def as_story_string(self, e2e: bool = True) -> Text:
        """Returns the story equivalent representation."""
        return super().as_story_string(e2e=True)


class WronglyClassifiedUserUtterance(UserUttered):
    """The NLU model predicted the wrong user utterance.

    Mostly used to mark wrong predictions and be able to
    dump them as stories."""

    type_name = "wrong_utterance"

    def __init__(self, event: UserUttered, eval_store: EvaluationStore) -> None:
        """Set `predicted_intent` and `predicted_entities` attributes."""
        try:
            self.predicted_intent = eval_store.intent_predictions[0]
        except LookupError:
            self.predicted_intent = None

        self.predicted_entities = eval_store.entity_predictions

        intent = {"name": eval_store.intent_targets[0]}

        super().__init__(
            event.text,
            intent,
            eval_store.entity_targets,
            event.parse_data,
            event.timestamp,
            event.input_channel,
        )

    def inline_comment(self) -> Optional[Text]:
        """A comment attached to this event. Used during dumping."""
        from rasa.shared.core.events import format_message

        if self.predicted_intent != self.intent["name"]:
            predicted_message = format_message(
                self.text, self.predicted_intent, self.predicted_entities
            )

            return f"predicted: {self.predicted_intent}: {predicted_message}"
        else:
            return None

    @staticmethod
    def inline_comment_for_entity(
        predicted: Dict[Text, Any], entity: Dict[Text, Any]
    ) -> Optional[Text]:
        """Returns the predicted entity which is then printed as a comment."""
        if predicted["entity"] != entity["entity"]:
            return "predicted: " + predicted["entity"] + ": " + predicted["value"]
        else:
            return None

    def as_story_string(self, e2e: bool = True) -> Text:
        """Returns text representation of event."""
        from rasa.shared.core.events import format_message

        correct_message = format_message(
            self.text, self.intent.get("name"), self.entities
        )
        return (
            f"{self.intent.get('name')}: {correct_message}   "
            f"<!-- {self.inline_comment()} -->"
        )


async def _create_data_generator(
    resource_name: Text,
    agent: "Agent",
    max_stories: Optional[int] = None,
    use_conversation_test_files: bool = False,
) -> "TrainingDataGenerator":
    from rasa.shared.core.generator import TrainingDataGenerator
    from rasa.shared.constants import DEFAULT_DOMAIN_PATH
    from rasa.model import get_model_subdirectories

    core_model = None
    if agent.model_directory:
        core_model, _ = get_model_subdirectories(agent.model_directory)

    if core_model and os.path.exists(os.path.join(core_model, DEFAULT_DOMAIN_PATH)):
        domain_path = os.path.join(core_model, DEFAULT_DOMAIN_PATH)
    else:
        domain_path = None

    test_data_importer = TrainingDataImporter.load_from_dict(
        training_data_paths=[resource_name], domain_path=domain_path
    )
    if use_conversation_test_files:
        story_graph = await test_data_importer.get_conversation_tests()
    else:
        story_graph = await test_data_importer.get_stories()

    return TrainingDataGenerator(
        story_graph,
        agent.domain,
        use_story_concatenation=False,
        augmentation_factor=0,
        tracker_limit=max_stories,
    )


def _clean_entity_results(
    text: Text, entity_results: List[Dict[Text, Any]]
) -> List["EntityPrediction"]:
    """Extract only the token variables from an entity dict."""
    cleaned_entities = []

    for r in tuple(entity_results):
        cleaned_entity = {ENTITY_ATTRIBUTE_TEXT: text}
        for k in (
            ENTITY_ATTRIBUTE_START,
            ENTITY_ATTRIBUTE_END,
            ENTITY_ATTRIBUTE_TYPE,
            ENTITY_ATTRIBUTE_VALUE,
        ):
            if k in set(r):
                if k == ENTITY_ATTRIBUTE_VALUE and EXTRACTOR in set(r):
                    # convert values to strings for evaluation as
                    # target values are all of type string
                    r[k] = str(r[k])
                cleaned_entity[k] = r[k]
        cleaned_entities.append(cleaned_entity)

    return cleaned_entities


def _get_full_retrieval_intent(parsed: Dict[Text, Any]) -> Text:
    """Return full retrieval intent, if it's present, or normal intent otherwise.

    Args:
        parsed: Predicted parsed data.

    Returns:
        The extracted intent.
    """
    base_intent = parsed.get(INTENT, {}).get(INTENT_NAME_KEY)
    response_selector = parsed.get(RESPONSE_SELECTOR, {})

    # return normal intent if it's not a retrieval intent
    if base_intent not in response_selector.get(
        RESPONSE_SELECTOR_RETRIEVAL_INTENTS, {}
    ):
        return base_intent

    # extract full retrieval intent
    # if the response selector parameter was not specified in config,
    # the response selector contains a "default" key
    if RESPONSE_SELECTOR_DEFAULT_INTENT in response_selector:
        full_retrieval_intent = (
            response_selector.get(RESPONSE_SELECTOR_DEFAULT_INTENT, {})
            .get(RESPONSE, {})
            .get(INTENT_RESPONSE_KEY)
        )
        return full_retrieval_intent if full_retrieval_intent else base_intent

    # if specified, the response selector contains the base intent as key
    full_retrieval_intent = (
        response_selector.get(base_intent, {})
        .get(RESPONSE, {})
        .get(INTENT_RESPONSE_KEY)
    )
    return full_retrieval_intent if full_retrieval_intent else base_intent


def _collect_user_uttered_predictions(
    event: UserUttered,
    predicted: Dict[Text, Any],
    partial_tracker: DialogueStateTracker,
    fail_on_prediction_errors: bool,
) -> EvaluationStore:
    user_uttered_eval_store = EvaluationStore()

    # intent from the test story, may either be base intent or full retrieval intent
    base_intent = event.intent.get(INTENT_NAME_KEY)
    full_retrieval_intent = event.intent.get(FULL_RETRIEVAL_INTENT_NAME_KEY)
    intent_gold = full_retrieval_intent if full_retrieval_intent else base_intent

    # predicted intent: note that this is only the base intent at this point
    predicted_base_intent = predicted.get(INTENT, {}).get(INTENT_NAME_KEY)

    # if the test story only provides the base intent AND the prediction was correct,
    # we are not interested in full retrieval intents and skip this section.
    # In any other case we are interested in the full retrieval intent (e.g. for report)
    if intent_gold != predicted_base_intent:
        predicted_base_intent = _get_full_retrieval_intent(predicted)

    user_uttered_eval_store.add_to_store(
        intent_targets=[intent_gold], intent_predictions=[predicted_base_intent]
    )

    entity_gold = event.entities
    predicted_entities = predicted.get(ENTITIES)

    if entity_gold or predicted_entities:
        user_uttered_eval_store.add_to_store(
            entity_targets=_clean_entity_results(event.text, entity_gold),
            entity_predictions=_clean_entity_results(event.text, predicted_entities),
        )

    if user_uttered_eval_store.check_prediction_target_mismatch():
        partial_tracker.update(
            WronglyClassifiedUserUtterance(event, user_uttered_eval_store)
        )
        if fail_on_prediction_errors:
            story_dump = YAMLStoryWriter().dumps(partial_tracker.as_story().story_steps)
            raise WrongPredictionException(
                f"NLU model predicted a wrong intent or entities. Failed Story:"
                f" \n\n{story_dump}"
            )
    else:
        end_to_end_user_utterance = EndToEndUserUtterance(
            event.text, event.intent, event.entities
        )
        partial_tracker.update(end_to_end_user_utterance)

    return user_uttered_eval_store


def emulate_loop_rejection(partial_tracker: DialogueStateTracker) -> None:
    """Add `ActionExecutionRejected` event to the tracker.

    During evaluation, we don't run action server, therefore in order to correctly
    test unhappy paths of the loops, we need to emulate loop rejection.

    Args:
        partial_tracker: a :class:`rasa.core.trackers.DialogueStateTracker`
    """
    from rasa.shared.core.events import ActionExecutionRejected

    rejected_action_name: Text = partial_tracker.active_loop_name
    partial_tracker.update(ActionExecutionRejected(rejected_action_name))


def _get_e2e_entity_evaluation_result(
    processor: "MessageProcessor",
    tracker: DialogueStateTracker,
    prediction: PolicyPrediction,
) -> Optional[EntityEvaluationResult]:
    previous_event = tracker.events[-1]
    if isinstance(previous_event, UserUttered):
        entities_predicted_by_policies = [
            entity
            for prediction_event in prediction.events
            if isinstance(prediction_event, EntitiesAdded)
            for entity in prediction_event.entities
        ]
        entity_targets = previous_event.entities
        if entity_targets or entities_predicted_by_policies:
            text = previous_event.text
            if text:
                parsed_message = processor.interpreter.featurize_message(
                    Message(data={TEXT: text})
                )
                if parsed_message:
                    tokens = parsed_message.get(TOKENS_NAMES[TEXT])
                    return EntityEvaluationResult(
                        entity_targets, entities_predicted_by_policies, tokens, text
                    )


def _run_action_prediction(
    processor: "MessageProcessor",
    partial_tracker: DialogueStateTracker,
    expected_action: Text,
) -> Tuple[Text, PolicyPrediction, EntityEvaluationResult]:
    action, prediction = processor.predict_next_action(partial_tracker)
    predicted_action = action.name()

    policy_entity_result = _get_e2e_entity_evaluation_result(
        processor, partial_tracker, prediction
    )

    if (
        prediction.policy_name
        and predicted_action != expected_action
        and _form_might_have_been_rejected(
            processor.domain, partial_tracker, predicted_action
        )
    ):
        # Wrong action was predicted,
        # but it might be Ok if form action is rejected.
        emulate_loop_rejection(partial_tracker)
        # try again
        action, prediction = processor.predict_next_action(partial_tracker)

        # Even if the prediction is also wrong, we don't have to undo the emulation
        # of the action rejection as we know that the user explicitly specified
        # that something else than the form was supposed to run.
        predicted_action = action.name()

    return predicted_action, prediction, policy_entity_result


def _collect_action_executed_predictions(
    processor: "MessageProcessor",
    partial_tracker: DialogueStateTracker,
    event: ActionExecuted,
    fail_on_prediction_errors: bool,
    circuit_breaker_tripped: bool,
) -> Tuple[EvaluationStore, PolicyPrediction, Optional[EntityEvaluationResult]]:
    from rasa.core.policies.form_policy import FormPolicy

    action_executed_eval_store = EvaluationStore()

    expected_action_name = event.action_name
    expected_action_text = event.action_text
    expected_action = expected_action_name or expected_action_text

    policy_entity_result = None
    prev_action_unlikely_intent = False

    if circuit_breaker_tripped:
        prediction = PolicyPrediction([], policy_name=None)
        predicted_action = "circuit breaker tripped"
    else:
        predicted_action, prediction, policy_entity_result = _run_action_prediction(
            processor, partial_tracker, expected_action
        )

    predicted_action_unlikely_intent = predicted_action == ACTION_UNLIKELY_INTENT_NAME
    if predicted_action_unlikely_intent and predicted_action != expected_action:
        partial_tracker.update(
            WronglyPredictedAction(
                predicted_action,
                expected_action_text,
                predicted_action,
                prediction.policy_name,
                prediction.max_confidence,
                event.timestamp,
                metadata=prediction.action_metadata,
            )
        )
        prev_action_unlikely_intent = True
        predicted_action, prediction, policy_entity_result = _run_action_prediction(
            processor, partial_tracker, expected_action
        )

    action_executed_eval_store.add_to_store(
        action_predictions=[predicted_action], action_targets=[expected_action]
    )

    if action_executed_eval_store.check_prediction_target_mismatch():
        partial_tracker.update(
            WronglyPredictedAction(
                expected_action_name,
                expected_action_text,
                predicted_action,
                prediction.policy_name,
                prediction.max_confidence,
                event.timestamp,
                metadata=prediction.action_metadata,
                predicted_action_unlikely_intent=prev_action_unlikely_intent,
            )
        )
        if (
            fail_on_prediction_errors
            and predicted_action != ACTION_UNLIKELY_INTENT_NAME
            and predicted_action != expected_action
        ):
            story_dump = YAMLStoryWriter().dumps(partial_tracker.as_story().story_steps)
            error_msg = (
                f"Model predicted a wrong action. Failed Story: " f"\n\n{story_dump}"
            )
            if FormPolicy.__name__ in prediction.policy_name:
                error_msg += (
                    "FormAction is not run during "
                    "evaluation therefore it is impossible to know "
                    "if validation failed or this story is wrong. "
                    "If the story is correct, add it to the "
                    "training stories and retrain."
                )
            raise WrongPredictionException(error_msg)
    elif prev_action_unlikely_intent:
        partial_tracker.update(
            WarningPredictedAction(
                ACTION_UNLIKELY_INTENT_NAME,
                predicted_action,
                prediction.policy_name,
                prediction.max_confidence,
                event.timestamp,
                prediction.action_metadata,
            )
        )
    else:
        partial_tracker.update(
            ActionExecuted(
                predicted_action,
                prediction.policy_name,
                prediction.max_confidence,
                event.timestamp,
                metadata=prediction.action_metadata,
            )
        )

    return action_executed_eval_store, prediction, policy_entity_result


def _form_might_have_been_rejected(
    domain: Domain, tracker: DialogueStateTracker, predicted_action_name: Text
) -> bool:
    return (
        tracker.active_loop_name == predicted_action_name
        and predicted_action_name in domain.form_names
    )


async def _predict_tracker_actions(
    tracker: DialogueStateTracker,
    agent: "Agent",
    fail_on_prediction_errors: bool = False,
    use_e2e: bool = False,
) -> Tuple[
    EvaluationStore,
    DialogueStateTracker,
    List[Dict[Text, Any]],
    List[EntityEvaluationResult],
]:

    processor = agent.create_processor()
    tracker_eval_store = EvaluationStore()

    events = list(tracker.events)

    partial_tracker = DialogueStateTracker.from_events(
        tracker.sender_id,
        events[:1],
        agent.domain.slots,
        sender_source=tracker.sender_source,
    )

    tracker_actions = []
    should_predict_another_action = True
    num_predicted_actions = 0
    policy_entity_results = []

    for event in events[1:]:
        if isinstance(event, ActionExecuted):
            circuit_breaker_tripped = processor.is_action_limit_reached(
                num_predicted_actions, should_predict_another_action
            )
            (
                action_executed_result,
                prediction,
                entity_result,
            ) = _collect_action_executed_predictions(
                processor,
                partial_tracker,
                event,
                fail_on_prediction_errors,
                circuit_breaker_tripped,
            )

            if entity_result:
                policy_entity_results.append(entity_result)

            if action_executed_result.action_targets:
                tracker_eval_store.merge_store(action_executed_result)
                tracker_actions.append(
                    {
                        "action": action_executed_result.action_targets[0],
                        "predicted": action_executed_result.action_predictions[0],
                        "policy": prediction.policy_name,
                        "confidence": prediction.max_confidence,
                    }
                )
                should_predict_another_action = processor.should_predict_another_action(
                    action_executed_result.action_predictions[0]
                )
                num_predicted_actions += 1

        elif use_e2e and isinstance(event, UserUttered):
            # This means that user utterance didn't have a user message, only intent,
            # so we can skip the NLU part and take the parse data directly.
            # Indirectly that means that the test story was in YAML format.
            if not event.text:
                predicted = event.parse_data
            # Indirectly that means that the test story was either:
            # in YAML format containing a user message, or in Markdown format.
            # Leaving that as it is because Markdown is in legacy mode.
            else:
                predicted = await processor.parse_message(UserMessage(event.text))
            user_uttered_result = _collect_user_uttered_predictions(
                event, predicted, partial_tracker, fail_on_prediction_errors
            )

            tracker_eval_store.merge_store(user_uttered_result)
        else:
            partial_tracker.update(event)
        if isinstance(event, UserUttered):
            num_predicted_actions = 0

    return tracker_eval_store, partial_tracker, tracker_actions, policy_entity_results


def _in_training_data_fraction(action_list: List[Dict[Text, Any]]) -> float:
    """Given a list of action items, returns the fraction of actions

    that were predicted using one of the Memoization policies."""
    from rasa.core.policies.ensemble import SimplePolicyEnsemble

    in_training_data = [
        a["action"]
        for a in action_list
        if a["policy"] and not SimplePolicyEnsemble.is_not_in_training_data(a["policy"])
    ]

    return len(in_training_data) / len(action_list) if action_list else 0


def _sort_trackers_with_severity_of_warning(
    trackers_to_sort: List[DialogueStateTracker],
) -> List[DialogueStateTracker]:
    """Sort the given trackers according to 'severity' of `action_unlikely_intent`.

    Severity is calculated by `IntentTEDPolicy` and is attached as
    metadata to `ActionExecuted` event.

    Args:
        trackers_to_sort: Trackers to be sorted

    Returns:
        Sorted trackers in descending order of severity.
    """
    tracker_severity_scores = []
    for tracker in trackers_to_sort:
        max_severity = 0
        for event in tracker.applied_events():
            if (
                isinstance(event, WronglyPredictedAction)
                and event.action_name_prediction == ACTION_UNLIKELY_INTENT_NAME
            ):
                max_severity = max(
                    max_severity,
                    event.metadata.get(QUERY_INTENT_KEY, {}).get(SEVERITY_KEY, 0),
                )
        tracker_severity_scores.append(max_severity)

    sorted_trackers_with_severity = sorted(
        zip(tracker_severity_scores, trackers_to_sort),
        # tuple unpacking is not supported in
        # python 3.x that's why it might look a bit weird
        key=lambda severity_tracker_tuple: -severity_tracker_tuple[0],
    )

    return [tracker for (_, tracker) in sorted_trackers_with_severity]


async def _collect_story_predictions(
    completed_trackers: List["DialogueStateTracker"],
    agent: "Agent",
    fail_on_prediction_errors: bool = False,
    use_e2e: bool = False,
) -> Tuple[StoryEvaluation, int, List[EntityEvaluationResult]]:
    """Test the stories from a file, running them through the stored model."""
    from sklearn.metrics import accuracy_score
    from tqdm import tqdm

    story_eval_store = EvaluationStore()
    failed_stories = []
    successful_stories = []
    stories_with_warnings = []
    correct_dialogues = []
    number_of_stories = len(completed_trackers)

    logger.info(f"Evaluating {number_of_stories} stories\nProgress:")

    action_list = []
    entity_results = []

    for tracker in tqdm(completed_trackers):
        (
            tracker_results,
            predicted_tracker,
            tracker_actions,
            tracker_entity_results,
        ) = await _predict_tracker_actions(
            tracker, agent, fail_on_prediction_errors, use_e2e
        )

        entity_results.extend(tracker_entity_results)

        story_eval_store.merge_store(tracker_results)

        action_list.extend(tracker_actions)

        if tracker_results.check_prediction_target_mismatch():
            # there is at least one wrong prediction
            failed_stories.append(predicted_tracker)
            correct_dialogues.append(0)
        else:
            successful_stories.append(predicted_tracker)
            correct_dialogues.append(1)

            if any(
                isinstance(event, WronglyPredictedAction)
                and event.action_name_prediction == ACTION_UNLIKELY_INTENT_NAME
                for event in predicted_tracker.events
            ):
                stories_with_warnings.append(predicted_tracker)

    logger.info("Finished collecting predictions.")

    in_training_data_fraction = _in_training_data_fraction(action_list)

    if len(correct_dialogues):
        accuracy = accuracy_score([1] * len(correct_dialogues), correct_dialogues)
    else:
        accuracy = 0

    _log_evaluation_table(
        [1] * len(completed_trackers),
        "END-TO-END" if use_e2e else "CONVERSATION",
        accuracy,
    )

    return (
        StoryEvaluation(
            evaluation_store=story_eval_store,
            failed_stories=failed_stories,
            successful_stories=successful_stories,
            stories_with_warnings=_sort_trackers_with_severity_of_warning(
                stories_with_warnings
            ),
            action_list=action_list,
            in_training_data_fraction=in_training_data_fraction,
        ),
        number_of_stories,
        entity_results,
    )


def _filter_step_events(step: StoryStep) -> StoryStep:
    events = []
    for event in step.events:
        if (
            isinstance(event, WronglyPredictedAction)
            and event.action_name
            == event.action_name_prediction
            == ACTION_UNLIKELY_INTENT_NAME
        ):
            continue
        events.append(event)
    updated_step = step.create_copy(use_new_id=False)
    updated_step.events = events
    return updated_step


def _log_stories(
    trackers: List[DialogueStateTracker], file_path: Text, message_if_no_trackers: Text
) -> None:
    """Write given stories to the given file."""
    with open(file_path, "w", encoding=DEFAULT_ENCODING) as f:
        if not trackers:
            f.write(f"# {message_if_no_trackers}")
        else:
            stories = [tracker.as_story(include_source=True) for tracker in trackers]
            steps = [
                _filter_step_events(step)
                for story in stories
                for step in story.story_steps
            ]
            f.write(YAMLStoryWriter().dumps(steps))


async def test(
    stories: Text,
    agent: "Agent",
    max_stories: Optional[int] = None,
    out_directory: Optional[Text] = None,
    fail_on_prediction_errors: bool = False,
    e2e: bool = False,
    disable_plotting: bool = False,
    successes: bool = False,
    errors: bool = True,
    warnings: bool = True,
) -> Dict[Text, Any]:
    """Run the evaluation of the stories, optionally plot the results.

    Args:
        stories: the stories to evaluate on
        agent: the agent
        max_stories: maximum number of stories to consider
        out_directory: path to directory to results to
        fail_on_prediction_errors: boolean indicating whether to fail on prediction
            errors or not
        e2e: boolean indicating whether to use end to end evaluation or not
        disable_plotting: boolean indicating whether to disable plotting or not
        successes: boolean indicating whether to write down successful predictions or
            not
        errors: boolean indicating whether to write down incorrect predictions or not
        warnings: boolean indicating whether to write down prediction warnings or not

    Returns:
        Evaluation summary.
    """
    from rasa.model_testing import get_evaluation_metrics

    generator = await _create_data_generator(stories, agent, max_stories, e2e)
    completed_trackers = generator.generate_story_trackers()

    story_evaluation, _, entity_results = await _collect_story_predictions(
        completed_trackers, agent, fail_on_prediction_errors, e2e
    )

    evaluation_store = story_evaluation.evaluation_store

    with pywarnings.catch_warnings():
        from sklearn.exceptions import UndefinedMetricWarning

        pywarnings.simplefilter("ignore", UndefinedMetricWarning)

        targets, predictions = evaluation_store.serialise()

        if out_directory:
            report, precision, f1, action_accuracy = get_evaluation_metrics(
                targets, predictions, output_dict=True
            )

            # Add conversation level accuracy to story report.
            num_failed = len(story_evaluation.failed_stories)
            num_correct = len(story_evaluation.successful_stories)
            num_warnings = len(story_evaluation.stories_with_warnings)
            num_convs = num_failed + num_correct
            if num_convs and isinstance(report, Dict):
                conv_accuracy = num_correct / num_convs
                report["conversation_accuracy"] = {
                    "accuracy": conv_accuracy,
                    "correct": num_correct,
                    "with_warnings": num_warnings,
                    "total": num_convs,
                }
            report_filename = os.path.join(out_directory, REPORT_STORIES_FILE)
            rasa.shared.utils.io.dump_obj_as_json_to_file(report_filename, report)
            logger.info(f"Stories report saved to {report_filename}.")

        else:
            report, precision, f1, action_accuracy = get_evaluation_metrics(
                targets, predictions, output_dict=True
            )

        evaluate_entities(
            entity_results,
            POLICIES_THAT_EXTRACT_ENTITIES,
            out_directory,
            successes,
            errors,
            disable_plotting,
        )

    telemetry.track_core_model_test(len(generator.story_graph.story_steps), e2e, agent)

    _log_evaluation_table(
        evaluation_store.action_targets,
        "ACTION",
        action_accuracy,
        precision=precision,
        f1=f1,
        in_training_data_fraction=story_evaluation.in_training_data_fraction,
    )

    if not disable_plotting and out_directory:
        _plot_story_evaluation(
            evaluation_store.action_targets,
            evaluation_store.action_predictions,
            out_directory,
        )

    if errors and out_directory:
        _log_stories(
            story_evaluation.failed_stories,
            os.path.join(out_directory, FAILED_STORIES_FILE),
            "None of the test stories failed - all good!",
        )
    if successes and out_directory:
        _log_stories(
            story_evaluation.successful_stories,
            os.path.join(out_directory, SUCCESSFUL_STORIES_FILE),
            "None of the test stories succeeded :(",
        )
    if warnings and out_directory:
        _log_stories(
            story_evaluation.stories_with_warnings,
            os.path.join(out_directory, STORIES_WITH_WARNINGS_FILE),
            "No warnings for test stories",
        )

    return {
        "report": report,
        "precision": precision,
        "f1": f1,
        "accuracy": action_accuracy,
        "actions": story_evaluation.action_list,
        "in_training_data_fraction": story_evaluation.in_training_data_fraction,
        "is_end_to_end_evaluation": e2e,
    }


def _log_evaluation_table(
    golds: List[Any],
    name: Text,
    accuracy: float,
    report: Optional[Dict[Text, Any]] = None,
    precision: Optional[float] = None,
    f1: Optional[float] = None,
    in_training_data_fraction: Optional[float] = None,
    include_report: bool = True,
) -> None:  # pragma: no cover
    """Log the sklearn evaluation metrics."""
    logger.info(f"Evaluation Results on {name} level:")
    logger.info(f"\tCorrect:          {int(len(golds) * accuracy)} / {len(golds)}")
    if f1 is not None:
        logger.info(f"\tF1-Score:         {f1:.3f}")
    if precision is not None:
        logger.info(f"\tPrecision:        {precision:.3f}")
    logger.info(f"\tAccuracy:         {accuracy:.3f}")
    if in_training_data_fraction is not None:
        logger.info(f"\tIn-data fraction: {in_training_data_fraction:.3g}")

    if include_report and report is not None:
        logger.info(f"\tClassification report: \n{report}")


def _plot_story_evaluation(
    targets: PredictionList,
    predictions: PredictionList,
    output_directory: Optional[Text],
) -> None:
    """Plot a confusion matrix of story evaluation."""
    from sklearn.metrics import confusion_matrix
    from sklearn.utils.multiclass import unique_labels
    from rasa.utils.plotting import plot_confusion_matrix

    confusion_matrix_filename = CONFUSION_MATRIX_STORIES_FILE
    if output_directory:
        confusion_matrix_filename = os.path.join(
            output_directory, confusion_matrix_filename
        )

    cnf_matrix = confusion_matrix(targets, predictions)

    plot_confusion_matrix(
        cnf_matrix,
        classes=unique_labels(targets, predictions),
        title="Action Confusion matrix",
        output_file=confusion_matrix_filename,
    )


async def compare_models_in_dir(
    model_dir: Text,
    stories_file: Text,
    output: Text,
    use_conversation_test_files: bool = False,
) -> None:
    """Evaluates multiple trained models in a directory on a test set.

    Args:
        model_dir: path to directory that contains the models to evaluate
        stories_file: path to the story file
        output: output directory to store results to
        use_conversation_test_files: `True` if conversation test files should be used
            for testing instead of regular Core story files.
    """
    number_correct = defaultdict(list)

    for run in rasa.shared.utils.io.list_subdirectories(model_dir):
        number_correct_in_run = defaultdict(list)

        for model in sorted(rasa.shared.utils.io.list_files(run)):
            if not model.endswith("tar.gz"):
                continue

            # The model files are named like <config-name>PERCENTAGE_KEY<number>.tar.gz
            # Remove the percentage key and number from the name to get the config name
            config_name = os.path.basename(model).split(PERCENTAGE_KEY)[0]
            number_of_correct_stories = await _evaluate_core_model(
                model,
                stories_file,
                use_conversation_test_files=use_conversation_test_files,
            )
            number_correct_in_run[config_name].append(number_of_correct_stories)

        for k, v in number_correct_in_run.items():
            number_correct[k].append(v)

    rasa.shared.utils.io.dump_obj_as_json_to_file(
        os.path.join(output, RESULTS_FILE), number_correct
    )


async def compare_models(
    models: List[Text],
    stories_file: Text,
    output: Text,
    use_conversation_test_files: bool = False,
) -> None:
    """Evaluates multiple trained models on a test set.

    Args:
        models: Paths to model files.
        stories_file: path to the story file
        output: output directory to store results to
        use_conversation_test_files: `True` if conversation test files should be used
            for testing instead of regular Core story files.
    """
    number_correct = defaultdict(list)

    for model in models:
        number_of_correct_stories = await _evaluate_core_model(
            model, stories_file, use_conversation_test_files=use_conversation_test_files
        )
        number_correct[os.path.basename(model)].append(number_of_correct_stories)

    rasa.shared.utils.io.dump_obj_as_json_to_file(
        os.path.join(output, RESULTS_FILE), number_correct
    )


async def _evaluate_core_model(
    model: Text, stories_file: Text, use_conversation_test_files: bool = False
) -> int:
    from rasa.core.agent import Agent

    logger.info(f"Evaluating model '{model}'")

    agent = Agent.load(model)
    generator = await _create_data_generator(
        stories_file, agent, use_conversation_test_files=use_conversation_test_files
    )
    completed_trackers = generator.generate_story_trackers()

    # Entities are ignored here as we only compare number of correct stories.
    story_eval_store, number_of_stories, _ = await _collect_story_predictions(
        completed_trackers, agent
    )
    failed_stories = story_eval_store.failed_stories
    return number_of_stories - len(failed_stories)
