import logging

from typing import Any, Dict, Optional, Text

from rasa.nlu.featurizers.featurzier import Featurizer

logger = logging.getLogger(__name__)


class NGramFeaturizer(Featurizer):
    def __init__(self, component_config: Optional[Dict[Text, Any]] = None):
        super(NGramFeaturizer, self).__init__(component_config)

        logger.warning(
            "DEPRECATION warning: Using `NGramFeaturizer` is deprecated. "
            "Please use `CountVectorsFeaturizer`."
        )
