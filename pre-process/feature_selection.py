import cPickle as pkl
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import  SelectKBest, chi2, mutual_info_classif

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


def get_ngram_scores(data_reader, scores_out, max_order, feature_selection_method, topK="all"):
    print "calculating {} scores for {}-grams...".format(feature_selection_method.func_name, max_order)

    vocab = None
    vocab_size = None
    scores = []

    count_vectorizer= CountVectorizer(vocabulary=vocab,
                                      max_features=vocab_size,
                                      ngram_range=(1, max_order))
    x_train = count_vectorizer.fit_transform(data_reader.train.x_doc)
    vocab = count_vectorizer.vocabulary_
    fs = SelectKBest(feature_selection_method, topK)
    fs_train = fs.fit_transform(x_train, data_reader.train.y)
    feature_names = count_vectorizer.get_feature_names()
    for i, score in enumerate(fs.scores_):
        scores.append((score, feature_names[i]))

    scores.sort(reverse=True)

    with open(scores_out, 'w') as out:
        [out.write('{},{}\n'.format(score[1], score[0])) for score in scores]


if __name__ == '__main__':
    dir_data = "/Users/ds/git/financial-topic-modeling/data/bpcorpus/"
    f_dataset_docs = dir_data + "lda_features_201705/corpus_bp_stock_cls.npz"
    unigram_chi_scores_out = dir_data + "lda_features_201705/chi-unigram-scores.csv"
    unigram_mi_scores_out = dir_data + "lda_features_201705/mi-unigram-scores.csv"
    bigram_chi_scores_out = dir_data + "lda_features_201705/chi-bigram-scores.csv"
    bigram_mi_scores_out = dir_data + "lda_features_201705/mi-bigram-scores.csv"


    data_reader = DataReader(dataset=f_dataset_docs)
    get_ngram_scores(data_reader, unigram_chi_scores_out, 1, feature_selection_method=chi2)
    get_ngram_scores(data_reader, unigram_mi_scores_out, 1, feature_selection_method=mutual_info_classif)
    get_ngram_scores(data_reader, bigram_chi_scores_out, 2, feature_selection_method=chi2, topK=57216)
    get_ngram_scores(data_reader, bigram_mi_scores_out, 2, feature_selection_method=mutual_info_classif, topK=57216)
