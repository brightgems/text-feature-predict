"""
prepare BOW/TFIDF features, generate SVM-style file for SVM
LR classifier
"""

import numpy
import sys
import cPickle as pkl

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

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

Features = ["BOW", "ngrams"]
param_grid_LR = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
                 'solver': ['liblinear', 'newton-cg', 'lbfgs']}

class Baselines:
    def __init__(self, data_reader, vocab=None, vocab_ngrams=None,
                 vocab_size=100000, ngram_order=3, ngram_num=100000, verbose=0):
        self.data_reader = data_reader
        self.vocab = vocab
        self.vocab_ngrams = vocab_ngrams
        self.vocab_size = vocab_size
        if self.vocab:
            self.vocab_size = len(self.vocab)
        self.ngrams_order = ngram_order  # max order of ngrams
        self.ngrams_num = ngram_num  # max number of ngrams to keep
        if self.vocab_ngrams:
            self.ngrams_num = len(self.vocab_ngrams)
        self.verbose = verbose
        self.x_train = None
        self.x_valid = None
        self.x_test = None
        self.predicted = None
        self.cls_model = None

    def run(self):
        def _train(feature, use_tfidf):
            eval("self.feature_" + feature)(use_tfidf=use_tfidf)
            if use_tfidf:
                print "==============================="
                print "feature: {};".format(feature+"-TFIDF")
                print "==============================="
            else:
                print "==============================="
                print "feature: {};".format(feature)
                print "==============================="
            self.cls_LR()
            self.get_top_features(N=50, feature=feature)

        for feature in Features:
            _train(feature, use_tfidf=False)
            _train(feature, use_tfidf=True)

    def run_tune(self):
        results = []

        for feature in Features:
            sys.stdout.flush()
            print "tuning:", feature
            eval("self.feature_" + feature)(use_tfidf=False)
            results.append(self.tune_LR())
            print "tuning:", feature + "-TFIDF"
            eval("self.feature_" + feature)(use_tfidf=True)
            results.append(self.tune_LR())

        cnt = 0
        for feature in Features:
            print "\n==============================="
            print "feature: {}".format(feature)
            print "==============================="
            print "test accuracy:", results[cnt][0]
            self.cls_model = results[cnt][1]
            self.get_top_features(N=50, feature=feature)
            cnt += 1
            print "\n==============================="
            print "feature: {}".format(feature+"-TFIDF")
            print "==============================="
            print "test accuracy:", results[cnt][0]
            self.cls_model = results[cnt][1]
            self.get_top_features(N=50, feature=feature)
            cnt += 1

    def feature_BOW(self, use_tfidf=False):
        bow_transformer = CountVectorizer(vocabulary=self.vocab,
                                          max_features=self.vocab_size)
        self.x_train = bow_transformer.fit_transform(self.data_reader.train.x)
        self.vocab = bow_transformer.vocabulary_
        bow_transformer_test = CountVectorizer(vocabulary=self.vocab)
        self.x_test = bow_transformer_test.fit_transform(self.data_reader.test.x)
        if len(self.data_reader.valid.x) > 0:
            self.x_valid = bow_transformer_test.fit_transform(self.data_reader.valid.x)
        if use_tfidf:
            tfidf_transformer = TfidfTransformer()
            self.x_train = tfidf_transformer.fit_transform(self.x_train)
            self.x_test = tfidf_transformer.fit_transform(self.x_test)
            if len(self.data_reader.valid.x) > 0:
                self.x_valid = tfidf_transformer.fit_transform(self.x_valid)

    def feature_ngrams(self, use_tfidf=False):
        ngrams_transfomer = CountVectorizer(vocabulary=self.vocab_ngrams,
                                            max_features=self.ngrams_num,
                                            ngram_range=(1, self.ngrams_order))
        self.x_train = ngrams_transfomer.fit_transform(self.data_reader.train.x)
        self.vocab_ngrams = ngrams_transfomer.vocabulary_
        ngrams_transfomer_test = CountVectorizer(vocabulary=self.vocab_ngrams)
        self.x_test = ngrams_transfomer_test.fit_transform(self.data_reader.test.x)
        if len(self.data_reader.valid.x) > 0:
            self.x_valid = ngrams_transfomer_test.fit_transform(self.data_reader.valid.x)
        if use_tfidf:
            tfidf_transformer = TfidfTransformer()
            self.x_train = tfidf_transformer.fit_transform(self.x_train)
            self.x_test = tfidf_transformer.fit_transform(self.x_test)
            if len(self.data_reader.valid.x) > 0:
                self.x_valid = tfidf_transformer.fit_transform(self.x_valid)

    def cls_LR(self):
        self.cls_model = LogisticRegression(C=1, solver='lbfgs',
                                            max_iter=500,
                                            verbose=self.verbose)
        self.cls_model.fit(self.x_train, self.data_reader.train.y)
        self.predicted = self.cls_model.predict(self.x_test)
        accuracy = accuracy_score(self.data_reader.test.y, self.predicted)
        print "\ttest accuracy:", accuracy

    def tune_LR(self):
        cv_model = GridSearchCV(LogisticRegression(penalty='l2', max_iter=500),
                                param_grid=param_grid_LR,
                                verbose=5)
        cv_model.fit(self.x_train, self.data_reader.train.y)
        print "======================================"
        print "best model params:", cv_model.best_params_
        accu = cv_model.score(self.x_test, self.data_reader.test.y)
        print "best test accuracy:", accu
        print "======================================"
        return (accu, cv_model.best_estimator_)


    def get_top_features(self, N=30, feature="BOW"):
        # reversed dictionary (idx -> term)
        vocab_reversed = dict()
        if feature == "BOW":
            for kk, vv in self.vocab.iteritems():
                vocab_reversed[vv] = kk
        else:
            for kk, vv in self.vocab_ngrams.iteritems():
                vocab_reversed[vv] = kk

        print "number of coefficients: {}".format(self.cls_model.coef_.shape[1])
        coef = list(self.cls_model.coef_[0])
        coef_idx = sorted(range(len(coef)), key=lambda k: coef[k], reverse=True)

        print "------------------------------------"
        print "top {} positive features:".format(N)
        print "------------------------------------"
        for i in range(N):
            print "\t{0}\t{1}".format(vocab_reversed[coef_idx[i]], coef[coef_idx[i]])
        print "\n------------------------------------"
        print "top {} negative features:".format(N)
        print "------------------------------------"
        for i in range(1, N+1):
            print "\t{0}\t{1}".format(vocab_reversed[coef_idx[-i]], coef[coef_idx[-i]])
        print "\n"

    def output_feature(self, path_feature, feature="BOW", delim=" "):

        def _output(filename, set="train"):
            f = open(filename, "w")
            dataset = eval("self.x_{}".format(set))
            label = eval("self.data_reader.{}.y".format(set))
            # fixme: problem here in output csr matrix
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

if __name__ == "__main__":
    dataset = "../../data/corpus/corpus_bow.npz"

    data_reader = DataReader(dataset=dataset)
    myModel = Baselines(data_reader=data_reader, vocab_size=50000, ngram_num=1000000, ngram_order=2, verbose=0)
    myModel.run_tune()