import argparse
import logging
import os
from pathlib import Path
from typing import Dict, Optional, Text, List

import rasa.shared.utils.io
import rasa.utils.io
from rasa.constants import NUMBER_OF_TRAINING_STORIES_FILE, PERCENTAGE_KEY
from rasa.shared.importers.importer import TrainingDataImporter

logger = logging.getLogger(__name__)


async def train_comparison_models(
    story_file: Text,
    domain: Text,
    output_path: Text = "",
    exclusion_percentages: Optional[List] = None,
    policy_configs: Optional[List] = None,
    runs: int = 1,
    additional_arguments: Optional[Dict] = None,
) -> None:
    """Trains multiple models for comparison of policies."""
    import rasa.model_training

    exclusion_percentages = exclusion_percentages or []
    policy_configs = policy_configs or []

    for r in range(runs):
        logging.info("Starting run {}/{}".format(r + 1, runs))

        for current_run, percentage in enumerate(exclusion_percentages, 1):
            for policy_config in policy_configs:
                config_name = os.path.splitext(os.path.basename(policy_config))[0]
                logging.info(
                    "Starting to train {} round {}/{} with {}% exclusion".format(
                        config_name, current_run, len(exclusion_percentages), percentage
                    )
                )

                await rasa.model_training.train_core(
                    domain,
                    policy_config,
                    stories=story_file,
                    output=str(Path(output_path, f"run_{r +1}")),
                    fixed_model_name=config_name + PERCENTAGE_KEY + str(percentage),
                    additional_arguments={
                        **additional_arguments,
                        "exclusion_percentage": percentage,
                    },
                )


def get_no_of_stories(story_file: Text, domain: Text) -> int:
    """Gets number of stories in a file."""
    importer = TrainingDataImporter.load_from_dict(
        domain_path=domain, training_data_paths=[story_file]
    )
    story_graph = importer.get_stories()
    return len(story_graph.story_steps)


async def do_compare_training(
    args: argparse.Namespace,
    story_file: Text,
    additional_arguments: Optional[Dict] = None,
) -> None:
    """Train multiple models for comparison of policies and dumps the result."""
    await train_comparison_models(
        story_file=story_file,
        domain=args.domain,
        output_path=args.out,
        exclusion_percentages=args.percentages,
        policy_configs=args.config,
        runs=args.runs,
        additional_arguments=additional_arguments,
    )
    no_stories = get_no_of_stories(args.stories, args.domain)

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
