from typing import List, Optional, Text
from rasa.shared.nlu.training_data.message import Message


def remove_unfeaturized_messages(
    messages: List[Message], attribute: Text, featurizers: Optional[List[Text]]
):
    """Removes messages that don't have required features.

    Some NLU components require messages to have specific features to
    make training and prediction possible. If we don't filter out messages
    which don't have needed features, it will lead to errors.

    Args:
        messages: List of messages we want to filter out.
        attribute: Message attribute.
        featurizers: Names of featurizers to consider.

    Returns:
        List of messages where unfeaturized messages are removed.
    """
    filtered_messages = [
        message
        for message in messages
        if message.features_present(attribute=attribute, featurizers=featurizers)
    ]
    return filtered_messages
