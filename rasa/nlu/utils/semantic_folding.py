import os
import re
import sys
from collections import defaultdict
from typing import Text, List, Any

import numpy as np
from tqdm import tqdm
from scipy.sparse import csr_matrix, save_npz, load_npz
from scipy.io import mmwrite
from sparse_som import *

from sklearn.feature_extraction.text import CountVectorizer
import json


def save_sparse_csr(filename, array):
    # note that .npz extension is added automatically
    np.savez(filename, data=array.data, indices=array.indices,
             indptr=array.indptr, shape=array.shape)


class SemanticFoldingGenerator:

    def __init__(self, width: int, height: int, density: float = 0.02):
        self._width = width
        self._height = height
        self._density = density
        self._som = None
        self._vocab = set()
        self._snippets = []
        self._snippet_data = None

        with open("/home/jem-mosig/rasa/rasa/rasa/nlu/utils/vocabulary.txt", "r", encoding="utf8") as f:
            vocabulary = f.read().splitlines()
            print(f"Vocabulary size: {len(vocabulary)}")

        self.vectorizer = CountVectorizer(lowercase=False, vocabulary=vocabulary)

    @staticmethod
    def _words_of_line(line: Text) -> List[Text]:
        return re.sub(
            # there is a space or an end of a string after it
            r"[^\w#@&]+(?=\s|$)|"
            # there is a space or beginning of a string before it
            # not followed by a number
            r"(\s|^)[^\w#@&]+(?=[^0-9\s])|"
            # not in between numbers and not . or @ or & or - or #
            # e.g. 10'000.00 or blabla@gmail.com
            # and not url characters
            r"(?<=[^0-9\s])[^\w._~:/?#\[\]()@!$&*+,;=-]+(?=[^0-9\s])",
            " ",
            line,
        ).split()

    @staticmethod
    def file_system_scan(f: Any, dir_name: Text, file_extension: Text = ""):
        for root, dirs, files in os.walk(dir_name):
            for file in files:
                if file.endswith(file_extension):
                    f(os.path.join(root, file))

    def _load_snippets_from_file(self, filename: Text) -> None:
        with open(filename, "r", encoding="utf8") as file:
            snippet = ""
            for line in file:
                if len(line.strip()) > 0:
                    snippet += line
                else:
                    if snippet:
                        self._snippets.append(snippet)
                    snippet = ""

    def _train_som(self, data: csr_matrix, num_iterations: int = 10**6) -> None:
        dump_file_name = "book_som.npy"
        _, input_dim = data.shape
        self._som = BSom(self.height, self.width, input_dim, topol=topology.HEXA)
        # if os.path.exists(dump_file_name):
        #     print("Loading codebook...")
        #     self._som.codebook = np.load(dump_file_name)   # som.codebook.dtype
        # else:
        self._som.train(data, cool=cooling.LINEAR)
        np.save(dump_file_name, self._som.codebook)

    def train(self):
        self.vectorizer.fit(self._snippets)
        self._snippet_data = self.vectorizer.transform(self._snippets)
        print(self._snippet_data.shape)
        print(type(self._snippet_data))
        # save_npz("snippets_encoded", self._snippet_data)
        # save_sparse_csr("snippets_encoded", self._snippet_data)
        mmwrite("snippets_encoded", self._snippet_data)
        self._train_som(self._snippet_data)

    def fingerprint(self, word: Text) -> np.ndarray:
        word_index = self.vectorizer.vocabulary_.get(word)
        if word_index is None:
            raise ValueError(f"The word '{word}' with index {word_index} is not part of the vocabulary.")

        # Find all snippets (rows) which contain the `word`
        rows, cols = self._snippet_data.nonzero()
        row_indices = np.where(cols == word_index)[0]
        snippet_indices = rows[row_indices]

        fingerprint = np.zeros((self.height, self.width), dtype=np.int)

        # For all these snippets, find their position on the map (best matching unit)
        for s in snippet_indices:
            sys.stdout = open(os.devnull, 'w')  # Block 'print'
            bmus = self._som.bmus(self._snippet_data[s])
            sys.stdout = sys.__stdout__
            for bmu in bmus:
                fingerprint[bmu[0], bmu[1]] += 1

        # Apply threshold
        max_active = round(self._density * self.width * self.height)
        max_inactive = self.width * self.height - max_active
        fingerprint = np.reshape(
            fingerprint,
            (self.width * self.height, )
        )
        min_indices = np.argpartition(
            fingerprint,
            max_inactive
        )[:max_inactive]
        fingerprint[min_indices] = 0

        # fingerprint = np.reshape(fingerprint, (self.width, self.height))

        return fingerprint.nonzero()[0]

    @property
    def width(self):
        return self._width

    @property
    def height(self):
        return self._height


if __name__ == '__main__':
    sfg = SemanticFoldingGenerator(8, 8)

    # traverse root directory, and list directories as dirs and files as files
    dir = "/home/jem-mosig/books/"
    sfg.file_system_scan(sfg._load_snippets_from_file, dir, ".txt")
    sfg.train()

    np.set_printoptions(threshold=np.inf)
    # import matplotlib.pyplot as plt

    with open("vocab_encoded.json", "w+") as file:
        json.dump(sfg.vectorizer.vocabulary_, file)

    print(str(sfg.vectorizer.vocabulary_))

    exit(0)

    with open("model.txt", "w+", encoding="utf8") as file:
        progress = 0
        vocabulary_size = len(sfg.vectorizer.vocabulary_)
        for word in sfg.vectorizer.vocabulary_.keys():
            file.write(f"\"{word}\": {sfg.fingerprint(word).tolist()}\n")
            file.flush()
            print(f"{round(100.0 * progress / vocabulary_size):.2f}% ({word})")

    # print(sfg.fingerprint("Rasa"))

    # query = "chatbot"
    # while query:
    #     print(f"Fingerprint of '{query}':")
    #     plt.matshow(sfg.fingerprint(query))
    #     plt.show()
    #     query = input("Term: ")

    # plt.matshow(sfg.fingerprint("Rasa"), )
    # plt.title("Rasa")
    # plt.matshow(sfg.fingerprint("chatbot"))
    # plt.title("chatbot")
    # plt.matshow(sfg.fingerprint("source"))
    # plt.title("source")
    # plt.matshow(sfg.fingerprint("help"))
    # plt.title("help")
    # plt.show()
