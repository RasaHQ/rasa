from typing import List, Type, Text, Optional, Dict, Any, Tuple
import os
import re
import numpy as np
import scipy.sparse

from rasa.nlu.components import Component
from rasa.shared.nlu.training_data.message import Message
from rasa.nlu.tokenizers.tokenizer import Tokenizer
from rasa.nlu.featurizers.featurizer import SparseFeaturizer
from rasa.nlu.tokenizers.tokenizer import Token
from rasa.shared.nlu.training_data.features import Features
from rasa.shared.nlu.training_data.training_data import TrainingData
from rasa.nlu.utils.semantic_map_utils import (
    SemanticMap,
    write_nlu_data_to_binary_file,
    run_smap,
)
from rasa.nlu.constants import (
    TOKENS_NAMES,
    DENSE_FEATURIZABLE_ATTRIBUTES,
    FEATURIZER_CLASS_ALIAS,
)
from rasa.shared.nlu.constants import (
    FEATURE_TYPE_SEQUENCE,
    FEATURE_TYPE_SENTENCE,
)
from rasa.utils.io import create_temporary_directory
import tempfile
import os


class SemanticMapFeaturizer(SparseFeaturizer):
    """Creates a sequence of sparse matrices based on a semantic map embedding."""

    @classmethod
    def required_components(cls) -> List[Type[Component]]:
        return [Tokenizer]

    @classmethod
    def required_packages(cls) -> List[Text]:
        return []

    defaults = {
        # filename of the pre-trained semantic map
        "semantic_map": None,
        # whether to convert all characters to lowercase
        "lowercase": True,
        # how to combine sequence features to a sentence feature
        "pooling": "sum",
    }

    def __init__(self, component_config: Optional[Dict[Text, Any]] = None) -> None:
        """Constructs a new semantic map vectorizer."""
        super().__init__(component_config)

        self.semantic_map_filename: Text = self.component_config["semantic_map"]
        self.lowercase: bool = self.component_config["lowercase"]
        self.pooling: Text = self.component_config["pooling"]

        if self.semantic_map_filename:
            if not os.path.exists(self.semantic_map_filename):
                raise FileNotFoundError(
                    f"Cannot find semantic map file '{self.semantic_map_filename}'"
                )
            self.semantic_map = SemanticMap(self.semantic_map_filename)
        else:
            self.semantic_map = None

        self._attributes = DENSE_FEATURIZABLE_ATTRIBUTES

    def train(self, training_data: TrainingData, *args: Any, **kwargs: Any,) -> None:
        """Converts tokens to features for training."""
        # Learn vocabulary and train semantic map
        if not self.semantic_map and len(training_data.nlu_examples) > 0:
            with tempfile.TemporaryDirectory() as temp_directory:
                if not os.path.exists(temp_directory):
                    raise FileNotFoundError(
                        f"Could not create temporary directory '{temp_directory}'."
                    )
                corpus_binary_filename = write_nlu_data_to_binary_file(
                    training_data, temp_directory
                )
                print(temp_directory)
                run_smap(temp_directory, corpus_binary_filename, 2, 2, epochs=2)
                print(os.listdir(os.path.join(temp_directory, "smap")))
                # TODO: CONTINUE HERE ...

        # Add features to be used by other components in the pipeline
        for example in training_data.training_examples:
            for attribute in self._attributes:
                self._set_semantic_map_features(example, attribute)

    def process(self, message: Message, **kwargs: Any) -> Optional[Dict[Text, Any]]:
        """Processes incoming message and compute and set features."""
        for attribute in self._attributes:
            self._set_semantic_map_features(message, attribute)

    def _set_semantic_map_features(self, message: Message, attribute: Text):
        sequence_features, sentence_features = self._featurize_tokens(
            message.get(TOKENS_NAMES[attribute], [])
        )

        if sequence_features is not None:
            final_sequence_features = Features(
                sequence_features,
                FEATURE_TYPE_SEQUENCE,
                attribute,
                self.component_config[FEATURIZER_CLASS_ALIAS],
            )
            message.add_features(final_sequence_features)

        if sentence_features is not None:
            final_sentence_features = Features(
                sentence_features,
                FEATURE_TYPE_SENTENCE,
                attribute,
                self.component_config[FEATURIZER_CLASS_ALIAS],
            )
            message.add_features(final_sentence_features)

    def _featurize_tokens(
        self, tokens: List[Token]
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Returns features of a list of tokens."""
        if not tokens:
            return None, None

        if self.lowercase:
            fingerprints = [
                self.semantic_map.get_term_fingerprint(token.text.lower())
                for token in tokens
            ]
        else:
            fingerprints = [
                self.semantic_map.get_term_fingerprint(token.text) for token in tokens
            ]

        sequence_features = scipy.sparse.vstack(
            [fp.as_coo_row_vector() for fp in fingerprints], "coo"
        )
        if self.pooling == "semantic_merge":
            sentence_features = self.semantic_map.semantic_merge(
                *fingerprints
            ).as_coo_row_vector()
        elif self.pooling == "mean":
            sentence_features = np.mean(sequence_features, axis=0)
        elif self.pooling == "sum":
            sentence_features = np.sum(sequence_features, axis=0)
        else:
            raise ValueError(
                f"Pooling operation '{self.pooling}' must be one of 'semantic_merge', 'mean', or 'sum"
            )

        assert sequence_features.shape == (
            len(fingerprints),
            self.semantic_map.num_cells,
        )
        assert sentence_features.shape == (1, self.semantic_map.num_cells)

        return sequence_features, sentence_features
