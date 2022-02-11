from rasa.core.featurizers.precomputation import MessageContainerForCoreFeaturization
from rasa.shared.core.domain import SubState
from typing import List, Optional, Dict, Set, Text


from rasa.shared.nlu.interpreter import NaturalLanguageInterpreter
from rasa.shared.nlu.training_data.features import Features
from rasa.shared.nlu.training_data.message import Message


class FeatureLookup:
    @classmethod
    def lookup_features(
        cls,
        message_data: SubState,
        precomputations: Optional[MessageContainerForCoreFeaturization],
        exclude_from_results: Optional[Set[Text]] = None,
    ) -> Dict[Text, List[Features]]:
        if precomputations is None:
            return {}
        exclude_from_results = exclude_from_results or {}
        attributes = set(
            attribute
            for attribute in message_data.keys()
            if attribute not in exclude_from_results
        )
        # Collect features for all those attributes
        output = precomputations.collect_features(message_data, attributes=attributes)
        return output
