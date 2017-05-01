"""
prepare BOW/TFIDF features, generate SVM-style file for SVM
LR classifier
"""

import numpy as np
import sys
import os
import textwrap
import cPickle as pkl
from scipy.sparse import hstack

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_selection import SelectKBest, chi2, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

def load_vocab(vocab_path):
    print "loading vocab ...",
    with open(vocab_path, "rb") as f:
        vocab = pkl.load(f)
    print "done", len(vocab), "words loaded!"
    return vocab

def load_lda(f_lda):
    with open(f_lda, "rb") as f:
        lda_features = pkl.load(f)
    return lda_features

class DataPoints:
    def __init__(self):
        self.x_doc = []
        self.x_stock = []
        self.x_lda = []
        self.y = []

    def set(self, data):
        self.x_doc = data[0]
        self.y = data[-1]
        if len(data) == 3:
            self.set_stock(data[1])

    def set_doc(self, x_doc):
        self.x_doc = x_doc

    def set_lda(self, x_lda):
        self.x_lda = np.array(x_lda)

    def set_stock(self, x_stock):
        self.x_stock = np.array(x_stock)

    def set_y(self, y):
        self.y = y

    def clear(self):
        self.x_doc = []
        self.x_stock = []
        self.x_lda = []
        self.y = []

class DataReader:
    """
    reader for logistic regression input
    data format: train, test, valid set
                 each set contains one list of documents (each doc as a list of words),
                 and one corresponding labels (integer 1/-1)
    """
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
        print "done! train:", len(self.train.y),\
              "valid:", len(self.valid.y), "test:", len(self.test.y)

class CustomSelectKBest(SelectKBest):
  """
    Extending SelectKBest with the ability to update a vocabulary that is given
    from a CountVectorizer object.
    Source: http://stackoverflow.com/questions/24939340/
    scikit-learn-update-countvectorizer-after-selecting-k-best-features
  """
  def __init__(self, score_func=f_classif, k=10):
    super(CustomSelectKBest, self).__init__(score_func, k)

  def transform_vocabulary(self, vocabulary):
    mask  = self.get_support(True)
    i_map = { j:i for i, j in enumerate(mask) }
    return { k:i_map[i] for k, i in vocabulary.iteritems() if i in i_map }

  def transform_vectorizer(self, cv):
    cv.vocabulary_ = self.transform_vocabulary(cv.vocabulary_)

Features = ["BOW", "ngrams"]
param_grid_LR = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
                 'solver': ['liblinear', 'newton-cg', 'lbfgs']}

class Baselines:
    def __init__(self, data_reader, vocab=None, vocab_ngrams=None,
                 vocab_size=100000, ngram_order=3, ngram_num=100000,
                 stock_today=False, stock_hist=20, f_lda=None,
                 verbose=0,
                 use_chi_square=None, top_k=10000000):
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
        self.stock_today = stock_today # whether to use today's stock change as feature
        self.stock_hist = stock_hist # number of historical stock change to include as features
        self.f_lda = f_lda # path to lda results (list)
        self.verbose = verbose
        self.x_train = None
        self.x_valid = None
        self.x_test = None
        self.predicted = None
        self.cls_model = None
        self.use_chi_square = use_chi_square
        self.top_k = top_k

    def run(self):
        def _get_feature(feature, use_tfidf):
            eval("self.feature_" + feature)(use_tfidf=use_tfidf)
            if use_tfidf:
                print "\tfeature: {}".format(feature+"-TFIDF"),
            else:
                print "\tfeature: {}".format(feature),
            self.cls_LR()
            #self.get_top_features(N=30, feature=feature)

        def _add_lda(feature, use_tfidf, lda_features):
            for topic_dist in lda_features:
                topic_num = topic_dist[0].shape[1]
                if use_tfidf:
                    print "\tfeature: {}-TFIDF-{}topic".format(feature, topic_num),
                else:
                    print "\tfeature: {}-{}topic".format(feature, topic_num),
                self.feature_lda(topic_dist)
                self.cls_LR()
                self.x_train = self.x_train[:, :-topic_num]
                self.x_test = self.x_test[:, :-topic_num]

        if self.f_lda:
            lda_features = load_lda(self.f_lda)

        for feature in Features:
            _get_feature(feature, use_tfidf=False)
            if self.f_lda:
                _add_lda(feature, use_tfidf=False, lda_features=lda_features)
            _get_feature(feature, use_tfidf=True)
            if self.f_lda:
                _add_lda(feature, use_tfidf=False, lda_features=lda_features)
            sys.stdout.flush()

    def run_tune(self):
        results = []

        for feature in Features:
            sys.stdout.flush()
            print "\ntuning:", feature
            eval("self.feature_" + feature)(use_tfidf=False)
            results.append(self.tune_LR())
            print "tuning:", feature + "-TFIDF"
            eval("self.feature_" + feature)(use_tfidf=True)
            results.append(self.tune_LR())
            sys.stdout.flush()

        cnt = 0

        print '============================================\nfinal results\n' \
              '============================================'
        for feature in Features:
            print "feature: {}".format(feature),
            print "\t[Accuracy] train:", results[cnt][1], "\ttest:", results[cnt][0]
            self.cls_model = results[cnt][2]
            # self.get_top_features(N=50, feature=feature)
            cnt += 1
            print "feature: {}".format(feature+"-TFIDF"),
            print "\t[Accuracy] train:", results[cnt][1], "\ttest:", results[cnt][0]
            self.cls_model = results[cnt][2]
            # self.get_top_features(N=50, feature=feature)
            cnt += 1
        sys.stdout.flush()

        print "\nstatistical info:"
        print "\tvocab size:", len(self.vocab)
        if self.vocab_ngrams:
            print "\tn-grams (n=2)", len(self.vocab_ngrams)

        print "\nvocabulary:", textwrap.fill(str(self.vocab), width=100)
        print "\nvocab-ngrams:", textwrap.fill(str(self.vocab_ngrams), width=100), "\n"

    def feature_lda(self, topic_dist):
        self.x_train = hstack([self.x_train, topic_dist[0]]).tocsr()
        self.x_test = hstack([self.x_test, topic_dist[1]]).tocsr()

    def feature_BOW(self, use_tfidf=False):
        bow_transformer = CountVectorizer(vocabulary=self.vocab,
                                          max_features=self.vocab_size)
        if self.use_chi_square is False:
            self.x_train = bow_transformer.fit_transform(self.data_reader.train.x_doc)
            self.vocab = bow_transformer.vocabulary_
        else:
            # feature selection over BOW
            term_doc = bow_transformer.fit_transform(self.data_reader.train.x_doc)
            chi_square = CustomSelectKBest(score_func=chi2, k=self.top_k)
            self.x_train = chi_square.fit_transform(term_doc, self.data_reader.train.y)
            self.vocab = chi_square.transform_vocabulary(bow_transformer.vocabulary_)

        bow_transformer_test = CountVectorizer(vocabulary=self.vocab)
        self.x_test = bow_transformer_test.fit_transform(self.data_reader.test.x_doc)
        if len(self.data_reader.valid.y) > 0:
            self.x_valid = bow_transformer_test.fit_transform(self.data_reader.valid.x_doc)
        if use_tfidf:
            tfidf_transformer = TfidfTransformer()
            self.x_train = tfidf_transformer.fit_transform(self.x_train)
            self.x_test = tfidf_transformer.fit_transform(self.x_test)
            if len(self.data_reader.valid.y) > 0:
                self.x_valid = tfidf_transformer.fit_transform(self.x_valid)

        # add stock price features
        if len(self.data_reader.train.x_stock) > 0 and self.stock_hist > 0:
            stock_start = 1
            stock_end = stock_start + self.stock_hist
            if self.stock_today:
                stock_start = 0
            self.x_train = hstack([self.x_train, self.data_reader.train.x_stock[:, stock_start:stock_end]]).tocsr()
            """
            ##### debug:NAN problem ######
            self.x_train = np.array(self.x_train.todense())
            print np.any(np.isnan(self.x_train))
            print np.argwhere(np.isnan(self.x_train))
            print np.argwhere(np.isnan(self.data_reader.train.x_stock))
            print np.all(np.isfinite(self.x_train))
            """
            self.x_test = hstack([self.x_test, self.data_reader.test.x_stock[:, stock_start:stock_end]]).tocsr()
            if len(self.data_reader.valid.y) > 0:
                self.x_valid = hstack([self.x_valid, self.data_reader.valid.x_stock[:, stock_start:stock_end]]).tocsr()

    def feature_ngrams(self, use_tfidf=False):
        ngrams_transfomer = CountVectorizer(vocabulary=self.vocab_ngrams,
                                            max_features=self.ngrams_num,
                                            ngram_range=(1, self.ngrams_order))

        if self.use_chi_square is False:
            self.x_train = ngrams_transfomer.fit_transform(self.data_reader.train.x_doc)
            self.vocab_ngrams = ngrams_transfomer.vocabulary_
        else:
            term_doc = ngrams_transfomer.fit_transform(self.data_reader.train.x_doc)
            chi_square = CustomSelectKBest(score_func=chi2, k=self.top_k)
            self.x_train = chi_square.fit_transform(term_doc, self.data_reader.train.y)
            self.vocab_ngrams = chi_square.transform_vocabulary(ngrams_transfomer.vocabulary_)

        ngrams_transfomer_test = CountVectorizer(vocabulary=self.vocab_ngrams)
        self.x_test = ngrams_transfomer_test.fit_transform(self.data_reader.test.x_doc)
        if len(self.data_reader.valid.y) > 0:
            self.x_valid = ngrams_transfomer_test.fit_transform(self.data_reader.valid.x_doc)
        if use_tfidf:
            tfidf_transformer = TfidfTransformer()
            self.x_train = tfidf_transformer.fit_transform(self.x_train)
            self.x_test = tfidf_transformer.fit_transform(self.x_test)
            if len(self.data_reader.valid.y) > 0:
                self.x_valid = tfidf_transformer.fit_transform(self.x_valid)

        # add stock price features
        if len(self.data_reader.train.x_stock) > 0 and self.stock_hist > 0:
            stock_start = 1
            stock_end = stock_start + self.stock_hist
            if self.stock_today:
                stock_start = 0
            self.x_train = hstack([self.x_train, self.data_reader.train.x_stock[:, stock_start:stock_end]]).tocsr()
            self.x_test = hstack([self.x_test, self.data_reader.test.x_stock[:, stock_start:stock_end]]).tocsr()
            if len(self.data_reader.valid.y) > 0:
                self.x_valid = hstack([self.x_valid, self.data_reader.valid.x_stock[:, stock_start:stock_end]]).tocsr()

    def cls_LR(self):
        self.cls_model = LogisticRegression(C=1, solver='lbfgs',
                                            max_iter=500,
                                            verbose=self.verbose)
        print '\t[feature num] {}'.format(self.x_train.shape[1]),
        self.cls_model.fit(self.x_train, self.data_reader.train.y)
        self.predicted = self.cls_model.predict(self.x_test)
        accu_train = self.cls_model.score(self.x_train, self.data_reader.train.y)
        accu = accuracy_score(self.data_reader.test.y, self.predicted)
        print "\t[Accuracy] train:", accu_train, "\ttest:", accu

    def tune_LR(self):
        cv_model = GridSearchCV(LogisticRegression(penalty='l2', max_iter=500),
                                param_grid=param_grid_LR,
                                verbose=5, return_train_score=True)
        cv_model.fit(self.x_train, self.data_reader.train.y)
        print "\n======================================"
        print "Tuning complete"
        print "Best score (on left out data):", cv_model.best_score_
        print "Tuning summary:"
        summary = ["split0_train_score", "split0_test_score",
                   "split1_train_score", "split1_test_score",
                   "split2_train_score", "split2_test_score"]
        for items in summary:
            print "\t{}: {}".format(items, cv_model.cv_results_[items])
        print "======================================\n"

        print "\n======================================"
        print "Best model:", cv_model.best_params_
        print "vocab size:", len(self.vocab)
        if self.vocab_ngrams:
            print "n-grams (n=2)", len(self.vocab_ngrams)
        print "train size:", self.x_train.shape
        print "test size:", self.x_test.shape
        accu_train = cv_model.score(self.x_train, self.data_reader.train.y)
        accu = cv_model.score(self.x_test, self.data_reader.test.y)
        print "[Accuracy] train:", accu_train, "\ttest:", accu
        print "======================================\n"
        return (accu, accu_train, cv_model.best_estimator_)


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
    '''
    # dir_data = "/home/yiren/Documents/time-series-predict/data/bp/"
    dir_data = "/Users/ds/git/financial-topic-modeling/data/bpcorpus/"
    f_dataset_docs = dir_data + "dataset/corpus_bp_cls.npz"

    data_reader = DataReader(dataset=f_dataset_docs)
    myModel = Baselines(data_reader=data_reader, ngram_num=1000000, ngram_order=2, verbose=0)
    myModel.run_tune()
    '''

    # Incremental number of features ###########################
    """
    dir_data = "/home/yiren/Documents/time-series-predict/data/bp/"
    f_dataset_docs = dir_data + "dataset/corpus_bp_stock_cls.npz"

    data_reader = DataReader(dataset=f_dataset_docs)


    vocab_top_k = [10, 100, 1000, 5000, 10000]
    for top_k in vocab_top_k:
        print 'performing classification for vocabulary size: {}'.format(top_k)
        for stock_hist in [0, 10, 20]:
            print "number of stock change hist: {}".format(stock_hist)
            myModel = Baselines(data_reader=data_reader, ngram_num=1000000, ngram_order=2,
                                stock_today=False, stock_hist=stock_hist,
                                verbose=0, use_chi_square=True, top_k=top_k)
            myModel.run_tune()
    """

    dir_data = "/Users/Irene/Documents/financial_topic_model/data/"
    dir_lda = dir_data + "lda/lda_results/"
    files = os.listdir(dir_lda)

    for path_lda in files:
        load_lda(dir_lda + path_lda)



