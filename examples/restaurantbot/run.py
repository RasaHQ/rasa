import argparse
import asyncio
import logging
from typing import Text

import os
import rasa.utils.io
import rasa.train
from examples.restaurantbot.policy import RestaurantPolicy
from rasa.core.agent import Agent
from rasa.core.policies.memoization import MemoizationPolicy
from rasa.core.policies.mapping_policy import MappingPolicy

logger = logging.getLogger(__name__)


async def parse(text: Text, model_path: Text):
    agent = Agent.load(model_path)

    response = await agent.handle_text(text)

    logger.info("Text: '{}'".format(text))
    logger.info("Response:")
    logger.info(response)

    return response


async def train_core(
    domain_file: Text = "domain.yml",
    model_directory: Text = "models",
    model_name: Text = "current",
    training_data_file: Text = "data/stories.md",
):
    agent = Agent(
        domain_file,
        policies=[
            MemoizationPolicy(max_history=3),
            MappingPolicy(),
            RestaurantPolicy(batch_size=100, epochs=100, validation_split=0.2),
        ],
    )

    training_data = await agent.load_data(training_data_file, augmentation_factor=10)
    agent.train(training_data)

    # Attention: agent.persist stores the model and all meta data into a folder.
    # The folder itself is not zipped.
    model_path = os.path.join(model_directory, model_name, "core")
    agent.persist(model_path)

    logger.info("Model trained. Stored in '{}'.".format(model_path))

    return model_path


def train_nlu(
    config_file="config.yml",
    model_directory: Text = "models",
    model_name: Text = "current",
    training_data_file="data/nlu.md",
):
    from rasa.nlu.training_data import load_data
    from rasa.nlu import config
    from rasa.nlu.model import Trainer

    training_data = load_data(training_data_file)
    trainer = Trainer(config.load(config_file))
    trainer.train(training_data)

    # Attention: trainer.persist stores the model and all meta data into a folder.
    # The folder itself is not zipped.
    model_path = os.path.join(model_directory, model_name)
    model_directory = trainer.persist(model_path, fixed_model_name="nlu")

    logger.info("Model trained. Stored in '{}'.".format(model_directory))

    return model_directory


if __name__ == "__main__":
    rasa.utils.io.configure_colored_logging(loglevel="INFO")

    parser = argparse.ArgumentParser(description="Restaurant Bot")

    subparser = parser.add_subparsers(dest="subparser_name")
    train_parser = subparser.add_parser("train", help="train a core or nlu model")
    parse_parser = subparser.add_parser("predict", help="predict next action")

    parse_parser.add_argument(
        "--model",
        default="models/current",
        help="Path to the model directory which contains "
        "sub-folders for core and nlu models.",
    )
    parse_parser.add_argument("--text", default="hello", help="Text to parse.")

    train_parser.add_argument(
        "model",
        choices=["nlu", "core"],
        help="Do you want to train a NLU or Core model?",
    )
    args = parser.parse_args()

    loop = asyncio.get_event_loop()

    # decide what to do based on first parameter of the script
    if args.subparser_name == "train":
        if args.model == "nlu":
            train_nlu()
        elif args.model == "core":
            loop.run_until_complete(train_core())
    elif args.subparser_name == "predict":
        loop.run_until_complete(parse(args.text, args.model))
