import logging
import os
import typing
from typing import Dict, Optional, Text, Union

from rasa.core.domain import Domain

if typing.TYPE_CHECKING:
    from rasa.core.interpreter import NaturalLanguageInterpreter

logger = logging.getLogger(__name__)


async def train(
    domain_file: Union[Domain, Text],
    stories_file: Text,
    output_path: Text,
    interpreter: Optional["NaturalLanguageInterpreter"] = None,
    endpoints: "AvailableEndpoints" = None,
    dump_stories: bool = False,
    policy_config: Text = None,
    exclusion_percentage: int = None,
    kwargs: Optional[Dict] = None,
):
    from rasa.core.agent import Agent
    from rasa.core import config, utils
    from rasa.core.utils import AvailableEndpoints

    if not endpoints:
        endpoints = AvailableEndpoints()

    if not kwargs:
        kwargs = {}

    policies = config.load(policy_config)

    agent = Agent(
        domain_file,
        generator=endpoints.nlg,
        action_endpoint=endpoints.action,
        interpreter=interpreter,
        policies=policies,
    )

    data_load_args, kwargs = utils.extract_args(
        kwargs,
        {
            "use_story_concatenation",
            "unique_last_num_states",
            "augmentation_factor",
            "remove_duplicates",
            "debug_plots",
        },
    )

    training_data = await agent.load_data(
        stories_file, exclusion_percentage=exclusion_percentage, **data_load_args
    )
    agent.train(training_data, **kwargs)
    agent.persist(output_path, dump_stories)

    return agent


async def train_comparison_models(
    stories,
    domain,
    output_path="",
    exclusion_percentages=None,
    policy_configs=None,
    runs=1,
    dump_stories=False,
    kwargs=None,
):
    """Train multiple models for comparison of policies"""
    from rasa.core import config

    exclusion_percentages = exclusion_percentages or []
    policy_configs = policy_configs or []

    for r in range(runs):
        logging.info("Starting run {}/{}".format(r + 1, runs))

        for i in exclusion_percentages:
            current_round = exclusion_percentages.index(i) + 1

            for policy_config in policy_configs:
                policies = config.load(policy_config)

                if len(policies) > 1:
                    raise ValueError(
                        "You can only specify one policy per model for comparison"
                    )

                policy_name = type(policies[0]).__name__
                output = os.path.join(
                    output_path, "run_" + str(r + 1), policy_name + str(current_round)
                )

                logging.info(
                    "Starting to train {} round {}/{}"
                    " with {}% exclusion"
                    "".format(policy_name, current_round, len(exclusion_percentages), i)
                )

                await train(
                    domain,
                    stories,
                    output,
                    policy_config=policy_config,
                    exclusion_percentage=i,
                    kwargs=kwargs,
                    dump_stories=dump_stories,
                )


async def get_no_of_stories(story_file, domain):
    """Get number of stories in a file."""
    from rasa.core.domain import TemplateDomain
    from rasa.core.training.dsl import StoryFileReader

    stories = await StoryFileReader.read_from_folder(
        story_file, TemplateDomain.load(domain)
    )
    return len(stories)


async def do_compare_training(cmdline_args, stories, additional_arguments):
    from rasa.core import utils

    await train_comparison_models(
        stories,
        cmdline_args.domain,
        cmdline_args.out,
        cmdline_args.percentages,
        cmdline_args.config,
        cmdline_args.runs,
        cmdline_args.dump_stories,
        additional_arguments,
    )

    no_stories = await get_no_of_stories(cmdline_args.stories, cmdline_args.domain)

    # store the list of the number of stories present at each exclusion
    # percentage
    story_range = [
        no_stories - round((x / 100.0) * no_stories) for x in cmdline_args.percentages
    ]

    story_n_path = os.path.join(cmdline_args.out, "num_stories.json")
    utils.dump_obj_as_json_to_file(story_n_path, story_range)


def do_interactive_learning(cmdline_args, stories, additional_arguments=None):
    from rasa.core.training import interactive

    if cmdline_args.core and cmdline_args.finetune:
        raise ValueError("--core can only be used without --finetune flag.")

    interactive.run_interactive_learning(
        stories,
        finetune=cmdline_args.finetune,
        skip_visualization=cmdline_args.skip_visualization,
        server_args=cmdline_args.__dict__,
        additional_arguments=additional_arguments,
    )


if __name__ == "__main__":
    raise RuntimeError(
        "Calling `rasa.core.train` directly is no longer supported. Please use "
        "`rasa train` to train a combined Core and NLU model or `rasa train core` "
        "to train a Core model."
    )
