import os
import re
from collections import defaultdict
from typing import Text, List, Any

import numpy as np
from scipy.sparse import csr_matrix, hstack
from sparse_som import *

from sklearn.feature_extraction.text import CountVectorizer


class SemanticFoldingGenerator:

    def __init__(self, width: int, height: int, density: float = 0.02):
        self._width = width
        self._height = height
        self._density = density
        self._som = None
        self._vocab = set()
        self._snippets = []
        self._snippet_data = None
        self.vectorizer = CountVectorizer(lowercase=False)

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
        _, input_dim = data.shape
        self._som = BSom(self.height, self.width, input_dim, topol=topology.HEXA, verbose=1)
        self._som.train(data, cool=cooling.LINEAR)

    def train(self):
        self.vectorizer.fit(self._snippets)
        self._snippet_data = self.vectorizer.transform(self._snippets)
        print(self._snippet_data.shape)
        self._train_som(self._snippet_data)

    def fingerprint(self, word: Text) -> np.ndarray:
        word_index = self.vectorizer.vocabulary_.get(word)
        if not word_index:
            raise ValueError("OOV")

        # Find all snippets (rows) which contain the `word`
        rows, cols = self._snippet_data.nonzero()
        row_indices = np.where(cols == word_index)[0]
        snippet_indices = rows[row_indices]

        fingerprint = np.zeros((self.height, self.width), dtype=np.int)

        # For all these snippets, find their position on the map (best matching unit)
        for s in snippet_indices:
            bmus = self._som.bmus(self._snippet_data[s])
            for bmu in bmus:
                fingerprint[bmu[0], bmu[1]] += 1

        # Apply threshold
        max_active = round(self._density * self.width * self.height)
        max_inactive = self.width * self.height - max_active
        print(max_active)
        fingerprint = np.reshape(
            fingerprint,
            (self.width * self.height, )
        )
        min_indices = np.argpartition(
            fingerprint,
            max_inactive
        )[:max_inactive]
        fingerprint[min_indices] = 0

        fingerprint = np.reshape(fingerprint, (self.width, self.height))

        return fingerprint

    @property
    def width(self):
        return self._width

    @property
    def height(self):
        return self._height


if __name__ == '__main__':
    sfg = SemanticFoldingGenerator(32, 32)

    # traverse root directory, and list directories as dirs and files as files
    dir = "/home/jem-mosig/rasa/rasa/docs/"
    sfg.file_system_scan(sfg._load_snippets_from_file, dir, ".rst")
    sfg.train()

    np.set_printoptions(threshold=np.inf)
    import matplotlib.pyplot as plt


    # print(sfg.fingerprint("chatbot"))
    # Display matrix

    query = "chatbot"
    while query:
        print(f"Fingerprint of '{query}':")
        plt.matshow(sfg.fingerprint(query))
        plt.show()
        query = input("Term: ")
