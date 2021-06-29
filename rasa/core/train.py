import argparse
import asyncio
import logging
import os
import tempfile
import typing
from typing import Dict, Optional, Text, Union, List

import rasa.shared.utils.io
import rasa.utils.io
from rasa.constants import NUMBER_OF_TRAINING_STORIES_FILE, PERCENTAGE_KEY
from rasa.core import training
from rasa.shared.core.constants import DEFAULT_ACTION_NAMES
from rasa.shared.core.domain import Domain
from rasa.shared.core.events import UserUttered, Event, ActionExecuted
from rasa.shared.core.trackers import DialogueStateTracker
from rasa.shared.core.training_data.structures import StoryGraph
from rasa.shared.importers.importer import TrainingDataImporter
from rasa.shared.nlu.constants import ACTION_NAME, ACTION_TEXT, INTENT, TEXT
from rasa.shared.nlu.training_data.message import Message
from rasa.shared.nlu.training_data.training_data import TrainingData
from rasa.utils.common import TempDirectoryPath

if typing.TYPE_CHECKING:
    from rasa.shared.nlu.interpreter import NaturalLanguageInterpreter
    from rasa.core.utils import AvailableEndpoints
    from rasa.core.agent import Agent

logger = logging.getLogger(__name__)


async def train(
    domain_file: Union[Domain, Text],
    training_resource: Union[Text, "TrainingDataImporter"],
    output_path: Text,
    interpreter: Optional["NaturalLanguageInterpreter"] = None,
    endpoints: "AvailableEndpoints" = None,
    policy_config: Optional[Union[Text, Dict]] = None,
    exclusion_percentage: Optional[int] = None,
    additional_arguments: Optional[Dict] = None,
    model_to_finetune: Optional["Agent"] = None,
) -> "Agent":
    from rasa.core import config, utils
    from rasa.core.utils import AvailableEndpoints
    from rasa.core.agent import Agent

    if not endpoints:
        endpoints = AvailableEndpoints()

    if not additional_arguments:
        additional_arguments = {}

    policies = config.load(policy_config)

    agent = Agent(
        domain_file,
        policies=policies,
        generator=endpoints.nlg,
        action_endpoint=endpoints.action,
        interpreter=interpreter,
    )

    data_load_args, additional_arguments = utils.extract_args(
        additional_arguments,
        {
            "use_story_concatenation",
            "unique_last_num_states",
            "augmentation_factor",
            "remove_duplicates",
            "debug_plots",
        },
    )
    logger.debug("Loading trackers")
    trackers, story_graph = await agent.load_data(
        training_resource, exclusion_percentage=exclusion_percentage, **data_load_args
    )
    story_to_training_data_converter = StoryToTrainingDataConverter()
    message_to_e2e_features_converter = MessageToE2EFeatureConverter()
    domain = domain_file
    logger.debug("Converting stories to training data")
    training_data = story_to_training_data_converter.convert_for_training(
        story_graph, domain
    )
    logger.debug("Featurizing messages")
    for msg in training_data.training_examples:
        interpreter.featurize_message(msg)
    logger.debug("Converting messages to e2e feature dict.")
    e2e_features = message_to_e2e_features_converter.convert(training_data)
    if model_to_finetune:
        agent.policy_ensemble = model_to_finetune.policy_ensemble
    logger.debug("Starting agent training")
    agent.train(trackers, e2e_features, **additional_arguments)
    agent.persist(output_path)

    return agent


async def train_comparison_models(
    story_file: Text,
    domain: Text,
    output_path: Text = "",
    exclusion_percentages: Optional[List] = None,
    policy_configs: Optional[List] = None,
    runs: int = 1,
    additional_arguments: Optional[Dict] = None,
) -> None:
    """Train multiple models for comparison of policies"""
    from rasa import model

    exclusion_percentages = exclusion_percentages or []
    policy_configs = policy_configs or []

    for r in range(runs):
        logging.info("Starting run {}/{}".format(r + 1, runs))

        for current_run, percentage in enumerate(exclusion_percentages, 1):
            for policy_config in policy_configs:

                file_importer = TrainingDataImporter.load_core_importer_from_config(
                    policy_config, domain, [story_file]
                )

                config_name = os.path.splitext(os.path.basename(policy_config))[0]
                logging.info(
                    "Starting to train {} round {}/{}"
                    " with {}% exclusion"
                    "".format(
                        config_name, current_run, len(exclusion_percentages), percentage
                    )
                )

                with TempDirectoryPath(tempfile.mkdtemp()) as train_path:
                    _, new_fingerprint = await asyncio.gather(
                        train(
                            domain,
                            file_importer,
                            train_path,
                            policy_config=policy_config,
                            exclusion_percentage=percentage,
                            additional_arguments=additional_arguments,
                        ),
                        model.model_fingerprint(file_importer),
                    )

                    output_dir = os.path.join(output_path, "run_" + str(r + 1))
                    model_name = config_name + PERCENTAGE_KEY + str(percentage)
                    model.package_model(
                        fingerprint=new_fingerprint,
                        output_directory=output_dir,
                        train_path=train_path,
                        fixed_model_name=model_name,
                    )


async def get_no_of_stories(story_file: Text, domain: Text) -> int:
    """Get number of stories in a file."""
    from rasa.shared.core.domain import Domain
    from rasa.shared.core.training_data import loading

    stories = await loading.load_data_from_files([story_file], Domain.load(domain))
    return len(stories)


async def do_compare_training(
    args: argparse.Namespace,
    story_file: Text,
    additional_arguments: Optional[Dict] = None,
) -> None:
    _, no_stories = await asyncio.gather(
        train_comparison_models(
            story_file=story_file,
            domain=args.domain,
            output_path=args.out,
            exclusion_percentages=args.percentages,
            policy_configs=args.config,
            runs=args.runs,
            additional_arguments=additional_arguments,
        ),
        get_no_of_stories(args.stories, args.domain),
    )

    # store the list of the number of stories present at each exclusion
    # percentage
    story_range = [
        no_stories - round((x / 100.0) * no_stories) for x in args.percentages
    ]

    training_stories_per_model_file = os.path.join(
        args.out, NUMBER_OF_TRAINING_STORIES_FILE
    )
    rasa.shared.utils.io.dump_obj_as_json_to_file(
        training_stories_per_model_file, story_range
    )


def do_interactive_learning(
    args: argparse.Namespace, file_importer: TrainingDataImporter
) -> None:
    from rasa.core.training import interactive

    interactive.run_interactive_learning(
        file_importer=file_importer,
        skip_visualization=args.skip_visualization,
        conversation_id=args.conversation_id,
        server_args=args.__dict__,
    )


class StoryToTrainingDataConverter:
    def convert_for_training(
        self, story_graph: StoryGraph, domain: Domain
    ) -> TrainingData:
        user_actions = [
            ActionExecuted(user_action) for user_action in domain.user_actions
        ]
        end_to_end_actions = [
            ActionExecuted(action_text=action_text)
            for action_text in domain.action_texts
        ]
        default_actions = [
            ActionExecuted(default_action) for default_action in DEFAULT_ACTION_NAMES
        ]

        user_events = []
        for step in story_graph.story_steps:
            user_events += [
                event for event in step.events if isinstance(event, UserUttered)
            ]

        messages = self._convert_tracker_to_messages(
            user_actions + end_to_end_actions + default_actions + user_events
        )
        # Workaround: add at least one end to end message to initialize
        # the `CountVectorizer` for e2e. Alternatives: Store information or simply config
        messages.append(
            Message(
                data=UserUttered(
                    text="hi", use_text_for_featurization=True
                ).as_sub_state()
            )
        )
        return TrainingData(training_examples=messages)

    def _convert_tracker_to_messages(self, events: List[Event]) -> List[Message]:
        messages = []
        for event in events:
            if isinstance(event, ActionExecuted):
                messages.append(Message(data=event.as_sub_state()))

            if isinstance(event, UserUttered):
                if event.use_text_for_featurization is None:
                    event.use_text_for_featurization = False
                    messages.append(Message(data=event.as_sub_state()))

                    event.use_text_for_featurization = True
                    messages.append(Message(data=event.as_sub_state()))

                    event.use_text_for_featurization = None
                else:
                    messages.append(Message(data=event.as_sub_state()))

        return messages

    def convert_for_inference(self, tracker: DialogueStateTracker) -> TrainingData:
        messages = self._convert_tracker_to_messages(tracker.events)

        return TrainingData(training_examples=messages)


class MessageToE2EFeatureConverter:
    def convert(self, training_data: TrainingData) -> Dict[Text, Message]:
        additional_features = {}
        for message in training_data.training_examples:
            texts = [
                v
                for k, v in message.data.items()
                if k in {ACTION_NAME, ACTION_TEXT, INTENT, TEXT}
            ]
            if len(texts) > 0:
                additional_features[texts[0]] = message

        return additional_features
