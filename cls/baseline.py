"""
prepare BOW/TFIDF features, generate SVM-style file for SVM
LR classifier
"""

import numpy
import cPickle as pkl

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

def load_vocab(vocab_path):
    print "loading vocab ...",
    with open(vocab_path, "rb") as f:
        vocab = pkl.load(f)
    print "done", len(vocab), "words loaded!"
    return vocab

class DataPoints:
    def __init__(self):
        self.x = []
        self.y = []

    def set(self, data):
        self.x = data[0]
        self.y = data[1]

    def set_x(self, x):
        self.x = x

    def set_y(self, y):
        self.y = y

    def clear(self):
        self.x = []
        self.y = []

class DataReader:
    def __init__(self, dataset):
        self.train = DataPoints()
        self.valid = DataPoints()
        self.test = DataPoints()
        self.load_data(dataset)

    def load_data(self, dataset_path):
        print "loading data ...",
        with open(dataset_path, "rb") as f:
            self.train.set(pkl.load(f))
            self.test.set(pkl.load(f))
            self.valid.set(pkl.load(f))
        print "done! train:", len(self.train.x),\
              "valid:", len(self.valid.x), "test:", len(self.test.x)

Features = ["BOW", "TFIDF"]
param_grid_NB = {'alpha': [0.01, 0.1, 1, 10]}
param_grid_LR = {'C': [0.01, 0.1, 1, 10, 100],
                 'solver': ['newton-cg', 'lbfgs', 'liblinear']}

class FeatureExtracor:
    def __init__(self, data_reader, vocab, vocab_ngrams=None,
                 ngram_order=3, ngram_num=100000, verbose=0):
        self.data_reader = data_reader
        self.vocab = vocab
        self.vocab_ngrams = vocab_ngrams
        self.ngrams_order = ngram_order  # max order of ngrams
        self.ngrams_num = ngram_num  # max number of ngrams to keep
        self.verbose = verbose
        self.x_train = None
        self.x_valid = None
        self.x_test = None
        self.predicted = None

    def feature_BOW(self, use_tfidf=False):
        bow_transformer = CountVectorizer(vocabulary=self.vocab)
        self.x_train = bow_transformer.fit_transform(self.data_reader.train.x)
        self.x_test = bow_transformer.fit_transform(self.data_reader.test.x)
        if len(self.data_reader.valid.x) > 0:
            self.x_valid = bow_transformer.fit_transform(self.data_reader.valid.x)
        if use_tfidf:
            tfidf_transformer = TfidfTransformer()
            self.x_train = tfidf_transformer.fit_transform(self.x_train)
            self.x_test = tfidf_transformer.fit_transform(self.x_test)
            if len(self.x_valid) > 0:
                self.x_valid = tfidf_transformer.fit_transform(self.x_valid)

    def feature_ngrams(self, use_tfidf=False):
        ngrams_transfomer = CountVectorizer(vocabulary=self.vocab_ngrams,
                                            max_features=self.ngrams_num,
                                            ngram_range=(1, self.ngrams_order))
        self.x_train = ngrams_transfomer.fit_transform(self.data_reader.train.x)
        ngrams_transfomer_test = CountVectorizer(vocabulary=ngrams_transfomer.vocabulary_)
        # bug fixed: keep the same vocab size for train and test
        self.x_test = ngrams_transfomer_test.fit_transform(self.data_reader.test.x)
        if len(self.data_reader.valid.x) > 0:
            self.x_valid = ngrams_transfomer_test.fit_transform(self.data_reader.valid.x)
        if use_tfidf:
            tfidf_transformer = TfidfTransformer()
            self.x_train = tfidf_transformer.fit_transform(self.x_train)
            self.x_test = tfidf_transformer.fit_transform(self.x_test)
            if len(self.x_valid) > 0:
                self.x_valid = tfidf_transformer.fit_transform(self.x_valid)

    def output_feature(self, path_feature, feature="BOW", delim=" "):
        def _output(filename, set="train"):
            f = open(filename, "w")
            dataset = eval("self.x_{}".format(set))
            label = eval("self.data_reader.{}.y".format(set))
            for lidx in xrange(len(dataset)):
                line = str(label[lidx])
                for fidx, feature in enumerate(dataset[lidx].split(delim)):
                    line += " {}:{}".format(fidx+1, feature)
                line += "\n"
                f.write(line)
            print "{}:{} written".format(set, len(dataset))

        _output("{}/{}.{}".format(path_feature, feature, "train"), set="train")
        _output("{}/{}.{}".format(path_feature, feature, "test"), set="test")
        if len(self.x_valid) > 0:
            _output("{}/{}.{}".format(path_feature, feature, "valid"), set="valid")

