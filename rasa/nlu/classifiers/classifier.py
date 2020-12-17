from typing import List, Optional, Any

from rasa.nlu.config import RasaNLUModelConfig
from rasa.nlu.components import Component
from rasa.utils.tensorflow.data_generator import DataChunkFile


class IntentClassifier(Component):
    """Intent classifier component."""

    def train_chunk(
        self,
        data_chunk_files: List[DataChunkFile],
        config: Optional[RasaNLUModelConfig] = None,
        **kwargs: Any,
    ) -> None:
        """Trains this component using the list of data chunk files.

        Args:
            data_chunk_files: List of data chunk files.
            config: The model configuration parameters.
        """
        pass
