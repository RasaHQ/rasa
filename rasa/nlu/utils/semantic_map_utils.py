from typing import Text, Set, Optional, Dict, List, BinaryIO, Tuple, Any
from scipy.sparse import csr_matrix, coo_matrix
import json
import numpy as np
import re
import subprocess
import os

from rasa.shared.nlu.training_data.training_data import TrainingData
from rasa.shared.nlu.constants import (
    INTENT,
    TEXT,
    INTENT_RESPONSE_KEY,
    ENTITIES,
    ENTITY_ATTRIBUTE_TYPE,
    ENTITY_ATTRIBUTE_VALUE,
)

from rasa.nlu.utils.semantic_map_converter import corpus_to_binary


class SemanticFingerprint:
    def __init__(self, height: int, width: int, activations: Set[int]) -> None:
        assert height > 0
        assert width > 0
        self.height = height
        self.width = width
        self.activations = activations

    def __len__(self):
        return len(self.activations)

    @property
    def num_cells(self):
        return self.height * self.width

    def as_ascii_art(self) -> Text:
        art = "\n"
        for row in range(self.height):
            for col in range(self.width):
                if col + self.width * row + 1 in self.activations:
                    art += "*"
                else:
                    art += " "
            art += "\n"
        return art

    def as_activations(self) -> Set[int]:
        return self.activations

    def as_csr_matrix(self, boost: Optional[float] = None) -> csr_matrix:
        if boost:
            data = [1.0 + boost * self._num_neightbours(a) for a in self.activations]
        else:
            data = np.ones(len(self.activations))
        row_indices = [(a - 1) // self.width for a in self.activations]
        col_indices = [(a - 1) % self.width for a in self.activations]

        return csr_matrix(
            (data, (row_indices, col_indices)),
            shape=(self.height, self.width),
            dtype=np.float32,
        )

    def as_coo_row_vector(self, boost: Optional[float] = None) -> coo_matrix:
        return self.as_csr_matrix(boost).reshape((1, -1)).tocoo()

    def as_dense_vector(self, boost: Optional[float] = None) -> np.array:
        return np.reshape(
            self.as_csr_matrix(boost).todense(), (self.height * self.width,)
        )

    def as_weighted_activations(
        self, boost: float = 1.0 / np.math.pi
    ) -> Dict[int, float]:
        return {a: 1.0 + boost * self._num_neightbours(a) for a in self.activations}

    def _num_neightbours(
        self, cell: int, local_topology: int = 8
    ) -> int:  # ToDo: Implement degree > 1 neighbourhood
        if local_topology == 4:
            return len(
                self.activations.intersection(
                    {
                        self._shift_onto_map(cell - 1),  # Left
                        self._shift_onto_map(cell + 1),  # Right
                        self._shift_onto_map(cell - self.width),  # Top
                        self._shift_onto_map(cell + self.width),  # Bottom
                    }
                )
            )
        elif local_topology == 8:
            return len(
                self.activations.intersection(
                    {
                        self._shift_onto_map(cell - 1),  # Left
                        self._shift_onto_map(cell + 1),  # Right
                        self._shift_onto_map(cell - self.width),  # Top
                        self._shift_onto_map(cell + self.width),  # Bottom
                        self._shift_onto_map(cell - 1 - self.width),  # Top Left
                        self._shift_onto_map(cell + 1 - self.width),  # Top Right
                        self._shift_onto_map(cell - 1 + self.width),  # Bottom Left
                        self._shift_onto_map(cell + 1 + self.width),  # Bottom Right
                    }
                )
            )
        else:
            raise ValueError(
                "Local topology must be either 4 or 8."
            )  # ToDo: Implement 6

    def _shift_onto_map(self, cell: int) -> int:
        """ Ensures that the given cell is on the map by translating its position """

        # Globally the map's topology is a torus, so
        # top and bottom edges are connected, and left
        # and right edges are connected.
        x = (cell - 1) % self.width
        y = (cell - 1) // self.width
        if y < 0:
            y += self.height * abs(y // self.height)
        return x + y * self.width + 1


class SemanticMap:
    def __init__(self, filename: Text) -> None:
        with open(filename, "r", encoding="utf-8") as file:
            data = json.load(file)

        self.width = data["Width"]
        self.height = data["Height"]
        self.local_topology = data["LocalTopology"]
        self.global_topology = data["GlobalTopology"]
        self.note = data["Note"]
        self._embeddings: Dict[Text, List[int]] = data["Embeddings"]
        self._vocab_pattern = re.compile(
            "|".join(
                [
                    r"\b" + re.escape(word) + r"\b"
                    for word in self._embeddings.keys()
                    if not word.startswith("<")
                ]
            )
        )
        self._special_token_pattern = re.compile(
            "|".join(
                [
                    re.escape(word)
                    for word in self._embeddings.keys()
                    if word.startswith("<")
                ]
            )
        )

    @property
    def num_cells(self):
        return self.height * self.width

    @property
    def vocabulary(self) -> Set[Text]:
        if not self._embeddings:
            return set()
        return set(self._embeddings.keys())

    def get_empty_fingerprint(self):
        return SemanticFingerprint(self.height, self.width, set())

    def get_term_fingerprint(self, term: Text) -> SemanticFingerprint:
        activations = self._embeddings.get(term.lower())
        if not activations:
            return self.get_empty_fingerprint()
        else:
            return SemanticFingerprint(self.height, self.width, set(activations))

    def get_term_fingerprint_as_csr_matrix(self, term: Text) -> SemanticFingerprint:
        return self.get_term_fingerprint(term).as_csr_matrix()

    def get_fingerprint(self, text: Text) -> SemanticFingerprint:
        term_fingerprints = [
            self.get_term_fingerprint(term) for term in self.get_known_terms(text)
        ]
        if term_fingerprints:
            return self.semantic_merge(*term_fingerprints)
        else:
            return self.get_empty_fingerprint()

    def get_known_terms(self, text: Text) -> List[Text]:
        terms = self._vocab_pattern.findall(
            text.lower()
        ) + self._special_token_pattern.findall(text.lower())
        return terms

    def semantic_merge(self, *fingerprints: SemanticFingerprint) -> SemanticFingerprint:
        if fingerprints:
            num_active = len(fingerprints[0])
            total = (
                np.sum([fp.as_csr_matrix(boost=0.3) for fp in fingerprints])
                .toarray()
                .flatten()
            )
            activations = np.argpartition(total, -num_active)[-num_active:] + 1
            return SemanticFingerprint(self.height, self.width, set(activations))
        else:
            return self.get_empty_fingerprint()

    def is_vocabulary_member(self, term: Text) -> bool:
        return term in self._embeddings

    def has_fingerprint(self, text: Text) -> bool:
        return len(self.get_known_terms(text.lower())) > 0

    @property
    def vocabulary(self) -> Set[Text]:
        return set(self._embeddings.keys())

    def as_dict(self) -> Dict[Text, Any]:
        return {
            "Width": self.width,
            "Height": self.height,
            "LocalTopology": self.local_topology,
            "GlobalTopology": self.global_topology,
            "Note": self.note,
            "Embeddings": self._embeddings,
        }


def _wrap_tag(kind: str, text: str) -> str:
    return f"[{kind}-{text}]"


def write_nlu_data_to_binary_file(
    nlu_data: TrainingData,
    dir_name: Text,
    use_intents: bool = True,
    lowercase: bool = True,
) -> Text:
    snippets = set()
    vocab = set()
    tag_vocab = set()
    for message in nlu_data.intent_examples:
        text_tokens = [token.text for token in message.get(f"{TEXT}_tokens", [])]
        intent = message.get(INTENT)
        intent_response_key = message.get(INTENT_RESPONSE_KEY)
        # entities = message.get(ENTITIES, [])

        if text_tokens:
            vocab.update(text_tokens)
            snippet = " ".join(text_tokens) + "\n"
        else:
            continue

        title = "= "
        if intent_response_key:
            word = _wrap_tag("intent", intent_response_key)
            tag_vocab.add(word)
            title += f"{word} "
        elif intent:
            word = _wrap_tag("intent", intent)
            tag_vocab.add(word)
            title += f"{word} "
        title += "=\n"

        if use_intents:
            snippets.add(title + snippet)
        else:
            snippets.add(snippet)

    with open(f"{dir_name}/smap.vocab.txt", "w") as file:
        file.writelines({v + "\n" for v in tag_vocab})
        file.writelines({v + "\n" for v in vocab})
    print(f"{len(tag_vocab)} tags")
    print(f"{len(vocab)} words")

    with open(f"{dir_name}/smap.corpus.txt", "w") as file:
        file.writelines(snippets)
    print(f"{len(snippets)} snippets")

    corpus_to_binary(
        output_filename=f"{dir_name}/smap.corpus.bin",
        data_directory_name=dir_name,
        data_filename_pattern="smap.corpus.txt",
        vocabulary_filename="smap.vocab.txt",
        enforce_lower_case=lowercase,
        max_processes=1,
        use_weights=True,
        combine_lists=False,
    )

    return f"{dir_name}/smap.vocab.txt", f"{dir_name}/smap.corpus.bin"


from subprocess import CalledProcessError, check_call, check_output


def run_smap(
    exe: Text,
    dir_name: Text,
    corpus_binary_filename: Text,
    height: int,
    width: int,
    epochs: int,
) -> Text:
    cmd = [
        exe,
        "create",
        corpus_binary_filename,
        str(height),
        str(width),
        "--directory",
        dir_name,
        "--name",
        "smap",
        "--epochs",
        str(epochs),
        "--local-topology",
        "6",
        "--global-topology",
        "0",
        "--non-adaptive",
    ]
    print(cmd)
    stdout = check_output(cmd)
    print(stdout.decode())
    return os.path.join(dir_name, "smap", "codebook.bin")


CODEBOOK_FILE_BYTEORDER = "little"


class SemanticMapCreator:
    def __init__(self, codebook_file_name: Text, vocabulary_file_name: Text):
        self.codebook: Optional[np.ndarray] = None
        self._norm: Optional[np.ndarray] = None
        self.vocabulary: Optional[List[Text]] = None

        if os.path.exists(vocabulary_file_name):
            self._load_vocabulary(vocabulary_file_name)
        else:
            raise FileNotFoundError(f"File {vocabulary_file_name} not found")

        if os.path.exists(codebook_file_name):
            self._load_codebook(codebook_file_name)
        else:
            raise FileNotFoundError(f"File {codebook_file_name} not found")

        assert len(self.vocabulary) == self.codebook.shape[2]

    def _load_codebook(self, filename: Text) -> None:
        with open(filename, "rb") as file:
            format_version: int = int.from_bytes(
                file.read(1), byteorder=CODEBOOK_FILE_BYTEORDER, signed=False
            )  # [uint8_t]
            assert format_version == 0

            height: int = int.from_bytes(
                file.read(8), byteorder=CODEBOOK_FILE_BYTEORDER, signed=False
            )  # [uint64_t]
            width: int = int.from_bytes(
                file.read(8), byteorder=CODEBOOK_FILE_BYTEORDER, signed=False
            )  # [uint64_t]
            input_dim: int = int.from_bytes(
                file.read(8), byteorder=CODEBOOK_FILE_BYTEORDER, signed=False
            )  # [uint64_t]

            num_entires = height * width * input_dim
            self.codebook = np.fromfile(file, dtype=np.float32)
        self.codebook.shape = (height, width, input_dim)
        self._norm = np.sum(self.codebook, axis=2).flatten()

    def _load_vocabulary(self, filename: Text) -> None:
        with open(filename, "r") as file:
            self.vocabulary = file.read().splitlines()

    def _fingerprint(
        self, index: int, max_density: float = 0.02, normalize: bool = True
    ) -> Tuple[Text, List[int]]:
        if not self.vocabulary or len(self.vocabulary) < index:
            raise ValueError(f"Vocabulary has no index {index}.")

        term = self.vocabulary[index]

        height, width, input_dim = self.codebook.shape
        assert 0 <= index
        assert index < input_dim

        num_active_cells: int = max(1, np.math.ceil(max_density * height * width))

        dense_fingerprint = self.codebook[:, :, index].flatten()
        if normalize:
            dense_fingerprint /= self._norm
        kth_largest_value = np.partition(dense_fingerprint, -num_active_cells)[
            -num_active_cells
        ]
        indices = np.add(
            np.argwhere(dense_fingerprint >= kth_largest_value).flatten(), 1
        ).tolist()
        print(indices)

        return term, indices

    @property
    def height(self) -> Optional[int]:
        if not self.codebook:
            return None
        height, _, _ = self.codebook.shape
        return height

    @property
    def width(self) -> Optional[int]:
        if not self.codebook:
            return None
        _, width, _ = self.codebook.shape
        return width

    def create_fingerprints(
        self, max_density: float = 0.02, normalize: bool = True, lowercase: bool = True,
    ) -> Dict[Text, List[int]]:
        fingerprints = dict()
        for vocabulary_index in range(len(self.vocabulary)):
            term, indices = self._fingerprint(
                vocabulary_index, max_density=max_density, normalize=normalize
            )
            if lowercase:
                term = term.lower()
            fingerprints[term] = indices
        return fingerprints


def semantic_overlap(
    fp1: SemanticFingerprint, fp2: SemanticFingerprint, method: Text = "Jaccard"
) -> float:
    """Returns the overlap score of the two fingerprints.

    The score is a floating point number between 0 and 1, where
    0 means that the two words are unrelated and 1 means that
    they share exactly the same meaning.
    """
    if method == "SzymkiewiczSimpson":
        return _szymkiewicz_simpson_overlap(fp1, fp2)
    elif method == "Jaccard":
        return _jaccard_overlap(fp1, fp2)
    elif method == "Rand":
        return _rand_overlap(fp1, fp2)
    else:
        raise ValueError(
            f"Method '{method}' is not one of 'SzymkiewiczSimpson', 'Jaccard', or 'Rand'"
        )


def _szymkiewicz_simpson_overlap(
    fp1: SemanticFingerprint, fp2: SemanticFingerprint
) -> float:
    num_common = len(fp1.as_activations().intersection(fp2.as_activations()))
    min_length = min(len(fp1.as_activations()), len(fp2.as_activations()))
    if min_length == 0:
        return 0
    else:
        return float(num_common / min_length)


def _jaccard_overlap(fp1: SemanticFingerprint, fp2: SemanticFingerprint) -> float:
    num_common = len(fp1.as_activations().intersection(fp2.as_activations()))
    union_length = len(fp1.as_activations().union(fp2.as_activations()))
    if union_length == 0:
        return 1.0
    else:
        return float(num_common / union_length)


def _rand_overlap(fp1: SemanticFingerprint, fp2: SemanticFingerprint) -> float:
    num_cells = fp1.height * fp1.width
    num_11 = len(fp1.as_activations().intersection(fp2.as_activations()))
    num_10 = len(fp1.as_activations().difference(fp2.as_activations()))
    num_01 = len(fp2.as_activations().difference(fp1.as_activations()))
    num_00 = num_cells - (num_10 + num_01 + num_11)
    return float((num_00 + num_11) / num_cells)
