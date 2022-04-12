from collections import Counter
from typing import Dict

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from leaderboard.utils import rasa_utils
from rasa.shared.nlu.training_data.training_data import TrainingData


def load_nlu_data(path_to_yaml: str) -> pd.DataFrame:
    """Load the nlu data into a dataframe where each column represents a message.

    Args:
        path_to_yaml: path to some nlu training data yaml
    Returns:
        dataframe with one row per message and columns corresponding to message
        properties (text, intent, entities)
    """
    return nlu_data_to_df(rasa_utils.load_nlu_data(path_to_yaml))


def nlu_data_to_df(nlu_data: TrainingData) -> pd.DataFrame:
    """Load the nlu data into a dataframe where each column represents a message.

    Args:
        nlu_data: some nlu training data
    Returns:
        dataframe with one row per message and columns corresponding to message
        properties (text, intent, entities)
    """
    return pd.DataFrame(
        {
            "text": [message.get("text") for message in nlu_data.nlu_examples],
            "intent": [message.get("intent") for message in nlu_data.nlu_examples],
            "entities": [
                [entity["entity"] for entity in message.get("entities", [])]
                for message in nlu_data.nlu_examples
            ],
        }
    )


def _sns_histplot(df_dict: Dict[str, pd.DataFrame], x: str, **kwargs) -> None:
    """... because pandas histchart is sometimes buggy."""
    common = pd.concat(list(df_dict.values()), keys=list(df_dict.keys()))
    common = common.reset_index()
    common = common.rename(columns={"level_0": "split"})
    _ = sns.histplot(common, x=x, multiple="dodge", hue="split", shrink=0.8, **kwargs)


def count_messages_per_entity(nlu_data_df: pd.DataFrame) -> pd.DataFrame:
    """Returns a dataframe listing the number of messages per entity.

    Args:
        nlu_data_df: dataframe created via `nlu_data_to_df`
    Returns:
        dataframe showing the `message_count` for each `entity` (type)
    """
    entities_counter = Counter()
    dedupliated = [list(set(entities)) for entities in nlu_data_df["entities"]]
    for entities in dedupliated:
        entities_counter.update(entities)

    entities_counter_df = pd.DataFrame.from_dict(
        entities_counter, orient="index"
    ).reset_index()
    entities_counter_df.columns = ["entity", "message_count"]
    entities_counter_df.sort_values("message_count", ascending=False, inplace=True)
    return entities_counter_df


def entities_per_message(nlu_data_df_dict: Dict[str, pd.DataFrame], **kwargs) -> None:
    """Analyse the number of messages that contain a certain entity, per entity.

    Plots a histogram and adds a new column `num_entities` to the given
    dataframe.

    Args:
        nlu_data_df_dict: dictionary mapping some name to dataframes loaded via
          `nlu_data_to_df`
    """
    for split_name, df in nlu_data_df_dict.items():
        df["num_entities"] = df["entities"].apply(len)

    # plot
    plt.figure(figsize=(10, 5))
    _sns_histplot(nlu_data_df_dict, x="num_entities", **kwargs)
    plt.xlabel("number of entities")
    plt.ylabel("number of messages")
    plt.title(
        "Histogram of the counts of entities in a message\n"
        "(Count the number of entities in a message - including duplicate entities)"
    )


def entity_support(nlu_data_df_dict: Dict[str, pd.DataFrame], **kwargs) -> None:
    """Analyse the number of messages that contain a certain entity, per entity.

    Plots a histogram and returns a dataframe with the respective counts.

    Args:
        nlu_data_df_dict: dictionary mapping some name to dataframes loaded via
          `nlu_data_to_df`
    Returns:
        dictionary of dataframes showing the `message_count` for each `entity` (type)
    """
    counts = {
        split: count_messages_per_entity(df) for split, df in nlu_data_df_dict.items()
    }

    # plot
    plt.figure(figsize=(10, 5))
    _sns_histplot(counts, x="message_count", **kwargs)
    plt.xlabel("number of messages")
    plt.ylabel("number of intents")
    plt.title(
        "Histogram of Entity Support Sizes\n"
        "(For each entity, count the number of messages that contain this entity)"
    )
    return counts


def intent_support(
    nlu_data_df_dict: Dict[str, pd.DataFrame], **kwargs
) -> Dict[str, pd.DataFrame]:
    """Analyse the number of messages that contain a certain intent, per entity.

    Plots a histogram and returns a dataframe with the respective counts.

    Args:
        nlu_data_df_dict: dictionary mapping some name to dataframes loaded via
          `nlu_data_to_df`
    Returns:
        dictionary of dataframes showing the `message_count` for each `intent`
    """
    support = dict()
    for split_name, df in nlu_data_df_dict.items():
        support[split_name] = nlu_data_df_dict[split_name]["intent"].value_counts()
        support[split_name] = pd.DataFrame(support[split_name]).reset_index()
        support[split_name].columns = ["intent", "message_count"]

    # plot
    plt.figure(figsize=(10, 5))
    _sns_histplot(support, x="message_count", **kwargs)
    plt.xlabel("number of messages")
    plt.ylabel("number of intents")
    plt.title(
        "Histogram of Intent Support Sizes\n"
        "(For each intent, count the number of messages that have this intent)"
    )
    return support
