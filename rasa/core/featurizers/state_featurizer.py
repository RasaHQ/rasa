import logging
from typing import (
    List,
    NamedTuple,
    Optional,
    Dict,
    Text,
)
from rasa.core import featurizers

from rasa.shared.core.domain import State, Domain
from rasa.shared.core.domain import Domain, Message
from rasa.shared.nlu.interpreter import NaturalLanguageInterpreter
from rasa.shared.nlu.constants import FEATURE_TYPE_SEQUENCE
from rasa.shared.nlu.training_data.features import Features
from rasa.core.featurizers import message_data_featurizer
from rasa.core.featurizers.message_data_featurizer import (
    MessageDataFeaturizerUsingInterpreter,
    MessageDataFeaturizerUsingMultiHotVectors,
    SetupMixin,
)

logger = logging.getLogger(__name__)


FEATURIZE_USING_INTERPRETER = "use_interpreter"
FEATURIZE_USING_MULTIHOT = "use_multihot"
POSTPROCESS_ENFORCE_SENTENCE_FEATURE = "enforce_sentence_feature"


class FeaturizationStep(NamedTuple):
    method: Text
    attributes: List[Text]


class FeaturizationConfig(NamedTuple):
    type: Text  # TODO: SubState -> MessageData
    pipeline: List[FeaturizationStep]
    postprocessing: List[FeaturizationStep]


class StateFeaturizer(SetupMixin):
    def __init__(
        self,
        config: List[FeaturizationConfig],
        featurizer_using_interpreter: Optional[
            MessageDataFeaturizerUsingInterpreter
        ] = None,
        featurizer_using_multihot: Optional[
            MessageDataFeaturizerUsingMultiHotVectors
        ] = None,
    ) -> None:
        self._config = {
            config_for_type.type: config_for_type for config_for_type in config
        }
        self._featurizers = {
            FEATURIZE_USING_INTERPRETER: featurizer_using_interpreter,
            FEATURIZE_USING_MULTIHOT: featurizer_using_multihot,
        }

    def setup(
        self, domain: Domain, interpreter: Optional[NaturalLanguageInterpreter]
    ) -> None:
        self.raise_if_setup_is(False)
        self._featurizers = {
            FEATURIZE_USING_INTERPRETER: MessageDataFeaturizerUsingInterpreter(
                interpreter=interpreter, domain=domain
            ),
            FEATURIZE_USING_MULTIHOT: MessageDataFeaturizerUsingMultiHotVectors(
                domain=domain
            ),
        }
        self._postprocessors = {
            POSTPROCESS_ENFORCE_SENTENCE_FEATURE: self._postprocess_features__autofill_sentence_feature,
        }

    def is_setup(self) -> bool:
        return (
            self._featurizer_using_interpreter.is_setup()
            and self._featurizer_using_multihot.is_setup()
        )

    def featurize(self, state: Dict[Text, Message]) -> Dict[Text, List[Features]]:
        self.raise_if_setup_is(False)

        attribute_feature_map = {}
        for type, message_data in state.items():

            config_for_type: FeaturizationConfig = self._config.get(type)

            feature_subset = dict()

            for step in config_for_type.pipeline:

                featurizer = featurizers.get(step.method)
                if featurizer is None:
                    raise ValueError(f"Unknown featurizer {step.method} requested.")
                new_features = featurizer.featurize(
                    message_data=message_data, attribute=step.attributes,
                )
                for attribute, feat in new_features:
                    feature_list = feature_subset.setdefault(attribute, [])
                    feature_list.append(feat)

            for step in config_for_type.postprocessing:

                postprocessor = featurizers.get(step.method)
                if postprocessor is None:
                    raise ValueError(f"Unknown postprocessor {step.method} requested.")
                postprocessor(
                    message_data=message_data,
                    feature_dict=feature_subset,
                    attribute=step.attributes,
                )

            attribute_feature_map.update(feature_subset)

        # NOTE: checking whether all expected attributes are populated here makes
        # no sense since that would require the interpreter to add some indicator
        # for "tried but didn't find any (e.g. entities)" here

        return attribute_feature_map

    def _postprocess_features__autofill_sentence_feature(
        self,
        message_data: Dict[Text, Any],
        feature_dict: Dict[Text, List[Features]],
        attributes: List[Text],
    ):
        """
        """
        for attribute in attributes:
            attribute_features = feature_dict.get(attribute, [])
            if any([f.type == FEATURE_TYPE_SEQUENCE for f in attribute_features]):
                # yeay, we have sentence features already
                # FIXME: this is different from current pipeline
                pass

            elif any([f.type == FEATURE_TYPE_SEQUENCE for f in attribute_features]):
                # we can deduce them from sequence features...
                sentence_features = message_data_featurizer.convert_sparse_sequence_to_sentence_features(
                    attribute_features
                )
            else:
                # we have nothing at all...
                fallback_feats = self._featurizer_using_multihot.featurize(
                    message_data=message_data, attribues=[attribute]
                )
                sentence_features = [fallback_feats[attribute]]

            feature_dict[attribute] = sentence_features
