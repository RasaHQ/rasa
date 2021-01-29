from typing import List, Type, Text, Optional, Dict, Any, Tuple
import os
import re
import json
import numpy as np
import scipy.sparse

from rasa.nlu.components import Component
from rasa.shared.nlu.training_data.message import Message
from rasa.nlu.tokenizers.tokenizer import Tokenizer
from rasa.nlu.featurizers.featurizer import SparseFeaturizer
from rasa.nlu.tokenizers.tokenizer import Token
from rasa.shared.nlu.training_data.features import Features
from rasa.nlu.model import Metadata
from rasa.shared.nlu.training_data.training_data import TrainingData
from rasa.nlu.utils.semantic_map_utils import (
    SemanticMap,
    SemanticMapCreator,
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
import rasa.utils.io


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
        # if no map name is given, the following parameters are relevant:
        # path to the semantic map training executable
        "executable": "/home/jem-mosig/projects/semantic-map/build/smap",
        # map size
        "height": 8,
        "width": 8,
        # number of training epochs
        "epochs": 10,
        # whether to add intent tags to snippets
        "use_intents": True,
        # whether to normalize fingerprints
        "normalize": True,
        # maximum density of any semantic fingerprint
        "max_density": 0.02,
        # whether to generate fingerprints with fixed density
        "fixed_density": False,
    }

    def __init__(self, component_config: Optional[Dict[Text, Any]] = None) -> None:
        """Constructs a new semantic map vectorizer."""
        super().__init__(component_config)

        self.semantic_map_filename: Optional[Text] = self.component_config[
            "semantic_map"
        ]

        self.lowercase: bool = self.component_config["lowercase"]
        self.pooling: Text = self.component_config["pooling"]

        self._size_for_training = (
            self.component_config["height"],
            self.component_config["width"],
        )
        self._epochs = self.component_config["epochs"]
        self._use_intents = self.component_config["use_intents"]
        self._normalize = self.component_config["normalize"]
        self._max_density = self.component_config["max_density"]
        self._executable: Optional[Text] = self.component_config["executable"]
        self.stats: Optional[Text] = None

        if self.semantic_map_filename:
            if not os.path.exists(self.semantic_map_filename):
                raise FileNotFoundError(
                    f"Cannot find semantic map file '{self.semantic_map_filename}'"
                )
            self.semantic_map = SemanticMap(self.semantic_map_filename)
        elif "semantic_map_data" in self.component_config:
            self.semantic_map = SemanticMap(
                data=self.component_config["semantic_map_data"]
            )
        else:
            self.semantic_map = None

        self._attributes = DENSE_FEATURIZABLE_ATTRIBUTES

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, type(self)):
            return False
        return (
            self.semantic_map
            and (
                self.semantic_map == other.semantic_map
                and self.pooling == other.pooling
                and self.lowercase == other.lowercase
            )
        ) or (
            not self.semantic_map
            and self.pooling == other.pooling
            and self.lowercase == other.lowercase
            and self._size_for_training == other._size_for_training
            and self._epochs == other._epochs
            and self._use_intents == other._use_intents
            and self._normalize == other._normalize
            and self._max_density == other._max_density
            and self._executable == other._executable
            and self.semantic_map_filename == other.semantic_map_filename
        )

    def persist(self, file_name: Text, model_dir: Text) -> Optional[Dict[Text, Any]]:
        """Persist this model into the passed directory.

        Returns the metadata necessary to load the model again.
        """

        file_name = file_name + ".pkl"

        if self.semantic_map:
            featurizer_file = os.path.join(model_dir, file_name)
            rasa.utils.io.json_pickle(
                featurizer_file,
                {
                    "pooling": self.pooling,
                    "lowercase": self.lowercase,
                    "stats": self.stats,
                    "semantic_map_data": self.semantic_map.as_dict(),
                },
            )

        return {"file": file_name}

    @classmethod
    def load(
        cls,
        meta: Dict[Text, Any],
        model_dir: Optional[Text] = None,
        model_metadata: Optional[Metadata] = None,
        cached_component: Optional["CountVectorsFeaturizer"] = None,
        **kwargs: Any,
    ) -> "SemanticMapFeaturizer":
        file_name = meta.get("file")
        featurizer_file = os.path.join(model_dir, file_name)

        if not os.path.exists(featurizer_file):
            return cls(meta)

        data = rasa.utils.io.json_unpickle(featurizer_file)
        return cls({**meta, **data})

    def train(self, training_data: TrainingData, *args: Any, **kwargs: Any,) -> None:
        """Converts tokens to features for training."""
        # Learn vocabulary and train semantic map
        if not self.semantic_map:
            if len(training_data.nlu_examples) == 0:
                raise ValueError("No nlu examples to train semantic map on.")
            with tempfile.TemporaryDirectory() as temp_directory:
                if not os.path.exists(temp_directory):
                    raise FileNotFoundError(
                        f"Could not create temporary directory '{temp_directory}'."
                    )
                (
                    vocabulary_filename,
                    corpus_binary_filename,
                ) = write_nlu_data_to_binary_file(
                    training_data,
                    temp_directory,
                    use_intents=self._use_intents,
                    lowercase=self.lowercase,
                )
                height, width = self._size_for_training
                epochs = self._epochs
                local_topology = 6
                global_topology = 0
                codebook_filename, convergence_stats_filename = run_smap(
                    self._executable,
                    temp_directory,
                    corpus_binary_filename,
                    height,
                    width,
                    epochs=epochs,
                )
                smc = SemanticMapCreator(codebook_filename, vocabulary_filename,)
                fps = smc.create_fingerprints(
                    max_density=self._max_density,
                    lowercase=self.lowercase,
                    normalize=self._normalize,
                )
                semantic_map_filename = os.path.join(temp_directory, "smap.json")
                with open(semantic_map_filename, "w") as file:
                    json.dump(
                        {
                            "Width": width,
                            "Height": height,
                            "LocalTopology": local_topology,
                            "GlobalTopology": global_topology,
                            "TrainingDataHash": training_data.fingerprint(),
                            "Note": "",
                            "Embeddings": fps,
                        },
                        file,
                    )
                self.semantic_map = SemanticMap(semantic_map_filename)
                self.stats = str(
                    rasa.shared.utils.io.read_file(convergence_stats_filename)
                )

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
            [fp.as_coo_row_vector(append_oov_feature=True) for fp in fingerprints],
            "coo",
        )
        if self.pooling == "semantic_merge":
            sentence_features = self.semantic_map.semantic_merge(
                *fingerprints
            ).as_coo_row_vector(append_oov_feature=True)
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
            self.semantic_map.num_cells + 1,
        )
        assert sentence_features.shape == (1, self.semantic_map.num_cells + 1)

        return sequence_features, sentence_features
