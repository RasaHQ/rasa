from mitie import *
import numpy as np
import urllib2
import os
import httplib
import multiprocessing


class MITIEFeaturizer(object):

    def __init__(self, fe_file):
        if not os.path.isfile(fe_file):
            self.download(fe_file)
            return  # need a cleaner solution for this
        self.feature_extractor = total_word_feature_extractor(fe_file)
        self.ndim = self.feature_extractor.num_dimensions

    def download_fe_file(self, fe_file):
        print("DOWNLOADING MITIE FILE")
        chunk_size = 30000000
        _fe_file = urllib2.urlopen("https://s3-eu-west-1.amazonaws.com/mitie/total_word_feature_extractor.dat")
        with open(fe_file, 'wb') as output:
            done = False
            while not done:
                print("fetching more...")
                try:
                    data = _fe_file.read(chunk_size)
                except httplib.IncompleteRead, e:
                    data = e.partial
                    done = True
                output.write(data)
        print("file written! {0}, {1}".format(fe_file, os.path.exists(fe_file)))

    def download(self, fe_file):
        download = multiprocessing.Process(target=self.download_fe_file, args=(fe_file,))
        download.start()

    def create_bow_vecs(self, sentences):
        X = np.zeros((len(sentences), self.ndim))

        for idx, sent in enumerate(sentences):
            tokens = tokenize(sent)
            vec = np.zeros(self.ndim)
            for token in tokens:
                vec += self.feature_extractor.get_feature_vector(token)
            X[idx, :] = vec / len(tokens)
        return X
