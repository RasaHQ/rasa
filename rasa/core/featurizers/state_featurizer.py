import logging
from typing import (
    List,
    NamedTuple,
    Optional,
    Dict,
    Text,
)
import scipy.sparse

from rasa.core import featurizers
from rasa.shared.core.domain import Domain, SubState
from rasa.shared.nlu.interpreter import NaturalLanguageInterpreter
from rasa.shared.nlu.constants import FEATURE_TYPE_SEQUENCE
from rasa.shared.nlu.training_data.features import Features
from rasa.shared.nlu.constants import FEATURE_TYPE_SENTENCE, FEATURE_TYPE_SEQUENCE
from rasa.core.featurizers.message_data_featurizer import (
    MessageDataFeaturizerUsingInterpreter,
    MessageDataFeaturizerUsingMultiHotVectors,
    SetupMixin,
)

logger = logging.getLogger(__name__)


FEATURIZE_USING_INTERPRETER = "use_interpreter"
FEATURIZE_USING_MULTIHOT = "use_multihot"
POSTPROCESS_ENFORCE_SENTENCE_FEATURE = "enforce_sentence_features"


class FeaturizationStep(NamedTuple):
    method: Text
    attributes: List[Text]


class FeaturizationConfig(NamedTuple):
    type: Text  # before: substate_type
    pipeline: List[FeaturizationStep]
    postprocessing: List[FeaturizationStep]


class StateFeaturizer(SetupMixin):
    def __init__(self, config: List[FeaturizationConfig],) -> None:
        self._config = {
            config_for_type.type: config_for_type for config_for_type in config
        }
        self._featurizers = None
        self._postprocessors = {
            POSTPROCESS_ENFORCE_SENTENCE_FEATURE: self._postprocess_features__autofill_sentence_feature,
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

    def is_setup(self) -> bool:
        return (
            self._featurizers is not None
            and self._featurizer_using_interpreter.is_setup()
            and self._featurizer_using_multihot.is_setup()
        )

    def featurize(self, state: Dict[Text, SubState]) -> Dict[Text, List[Features]]:
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
        message_data: SubState,
        feature_dict: Dict[Text, List[Features]],
        attributes: List[Text],
    ):
        for attribute in attributes:
            attribute_features = feature_dict.get(attribute, [])
            if any([f.type == FEATURE_TYPE_SEQUENCE for f in attribute_features]):
                # yeay, we have sentence features already
                # FIXME: this is different from current pipeline
                pass

            elif any([f.type == FEATURE_TYPE_SEQUENCE for f in attribute_features]):
                # we can deduce them from sequence features...
                sentence_features = convert_sparse_sequence_to_sentence_features(
                    attribute_features
                )
            else:
                # we have nothing at all...
                fallback_feats = self._featurizer_using_multihot.featurize(
                    message_data=message_data, attribues=[attribute]
                )
                sentence_features = [fallback_feats[attribute]]

            feature_dict[attribute] = sentence_features


def convert_sparse_sequence_to_sentence_features(
    features: List[Features],
) -> List[Features]:
    """Grabs all sparse sequence features and turns them into sparse sentence features.

    That is, this function filters all sparse sequence features and then
    turns each of these sparse sequence features into sparse sentence feature
    by summing over their first axis, respectively.

    TODO: move this somewhere else
    TODO: extend features to obtain meaningful aggregations

    Returns:
      a list with as many sparse sentenc features as there are sparse sequence features
      in the given list
    """
    # TODO: add functionality to `Features` to obtain a meaningful conversion from
    # sequence to sentence features depending on it's "origin" (requires rework of
    # Features class to work properly because the origin attribute is really a
    # tag defined by the user...)
    return [
        Features(
            scipy.sparse.coo_matrix(feature.features.sum(0)),
            FEATURE_TYPE_SENTENCE,
            feature.attribute,
            feature.origin,
        )
        for feature in features
        if (feature.is_sparse and feature.type == FEATURE_TYPE_SEQUENCE)
    ]
