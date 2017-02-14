from mitie import *
import urllib2
import os
import httplib
import multiprocessing
import progressbar
import logging


class MITIEFeaturizer(object):

    def __init__(self, fe_file):
        self.feature_extractor = total_word_feature_extractor(fe_file)
        self.ndim = self.feature_extractor.num_dimensions

    @staticmethod
    def download_fe_file(fe_file):
        logging.debug("DOWNLOADING MITIE FILE")
        chunk_size = 30000000
        _fe_file = urllib2.urlopen("https://s3-eu-west-1.amazonaws.com/mitie/total_word_feature_extractor.dat")
        _file_meta = _fe_file.info()
        file_size = int(_file_meta.getheaders("Content-Length")[0])
        print "Downloading: %s (%s MB)" % (fe_file, file_size/1024/1024)
        widgets = ['Progress: ',
                   progressbar.Percentage(),
                   ' ',
                   progressbar.Bar(marker='#', left='[', right=']')]
        bar = progressbar.ProgressBar(maxval=file_size, widgets=widgets)
        bar.start()
        bytes_read = 0
        with open(fe_file, 'wb') as output:
            done = False
            while not done:
                data = _fe_file.read(chunk_size)
                if not data:
                    done = True
                    bar.finish()
                else:
                    output.write(data)
                    bytes_read += len(data)
                    bar.update(bytes_read)
        logging.debug("file written! {0}, {1}".format(fe_file, os.path.exists(fe_file)))

    def create_bow_vecs(self, sentences):
        import numpy as np
        X = np.zeros((len(sentences), self.ndim))

        for idx, sent in enumerate(sentences):
            tokens = tokenize(sent)
            vec = np.zeros(self.ndim)
            for token in tokens:
                vec += self.feature_extractor.get_feature_vector(token)
            X[idx, :] = vec / len(tokens)
        return X
