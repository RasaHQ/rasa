import logging

from typing import Any, Dict, Optional, Text

from rasa.nlu.featurizers.featurizer import SparseFeaturizer

logger = logging.getLogger(__name__)


class NGramFeaturizer(SparseFeaturizer):
    def __init__(self, component_config: Optional[Dict[Text, Any]] = None) -> None:
        super(NGramFeaturizer, self).__init__(component_config)

        raise NotImplementedError(
            "REMOVAL warning: You cannot use `NGramFeaturizer` anymore. "
            "Please use `CountVectorsFeaturizer` instead. The following settings"
            "match the previous `NGramFeaturizer`:"
            ""
            "- name: 'CountVectorsFeaturizer'"
            "  analyzer: 'char_wb'"
            "  min_ngram: 3"
            "  max_ngram: 17"
            "  max_features: 10"
            "  min_df: 5"
        )
