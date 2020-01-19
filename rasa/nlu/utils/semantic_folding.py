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
        self._snippet_BOWs = []

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

    def _add_vocab_from_dir(self, dir_name: Text, file_extension: Text = ""):
        self.file_system_scan(self._add_vocab_from_file, dir_name, file_extension)

    def _add_vocab_from_file(self, filename: Text) -> None:
        with open(filename, "r", encoding="utf8") as file:
            for line in file:
                words = self._words_of_line(line)

                for word in words:
                    self._vocab.add(word)

    def _create_snippet_bows_from_dir(self, dir_name: Text, file_extension: Text = ""):
        self.file_system_scan(self._create_snippet_bows_from_file, dir_name, file_extension)

    def _create_snippet_bows_from_file(self, filename: Text, min_snippet_length: int = 10) -> None:
        with open(filename, "r", encoding="utf8") as file:
            bow = np.zeros(len(self._vocab))
            words_in_snippet = 0
            for line in file:
                if len(line.strip()) > 0:
                    words = self._words_of_line(line)
                    for word in words:
                        words_in_snippet += 1
                        for i, known_word in enumerate(self._vocab):
                            if word == known_word:
                                bow[i] += 1
                else:
                    if words_in_snippet > min_snippet_length:
                        self._snippet_BOWs.append(csr_matrix(bow, dtype=np.float32))
                    bow = np.zeros(len(self._vocab))
                    words_in_snippet = 0

    def _train_som(self, data: csr_matrix, num_iterations: int = 10**6) -> None:
        _, input_dim = data.shape
        self._som = BSom(self.height, self.width, input_dim, topol=topology.HEXA, verbose=1)
        self._som.train(data, tmax=num_iterations, cool=cooling.LINEAR)

    @property
    def width(self):
        return self._width

    @property
    def height(self):
        return self._height


if __name__ == '__main__':
    sfg = SemanticFoldingGenerator(12, 12)

    # traverse root directory, and list directories as dirs and files as files
    dir = "/home/jem-mosig/rasa/rasa/docs/"
    sfg._add_vocab_from_dir(dir, ".rst")
    sfg._create_snippet_bows_from_dir(dir, ".rst")

    print(len(sfg._snippet_BOWs))
